import torch
from torch import autocast, inference_mode
import torchvision.transforms as T
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
from utils import image_grid

from diffusers import (
    UNet2DModel,
    VQModel,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)

from utils import pil_to_tensor
from PIL import Image
from glob import glob
import copy
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Any, Dict, List, Optional, Tuple, Union
import logging



def forward_ResnetBlock2D(self, input_tensor, temb, inject_into_botleneck = None):
    ## From https://github.com/huggingface/diffusers/blob/fc94c60c8373862c509e388f3f4065d98cedf589/src/diffusers/models/resnet.py#L367
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.upsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)

    ### Middle of block injection
    bottleneck = hidden_states.clone()
    if not inject_into_botleneck is None:
        hidden_states = bottleneck + inject_into_botleneck

    hidden_states = self.nonlinearity(hidden_states) #### <---- 

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    # if self.conv_shortcut is not None:
    #     input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) 

    return output_tensor, bottleneck

def mid_block_forward(self, hidden_states, temb=None, encoder_states=None, inject_into_botleneck=None):
    ## https://github.com/huggingface/diffusers/blob/fc94c60c8373862c509e388f3f4065d98cedf589/src/diffusers/models/unet_2d_blocks.py#L246
    hidden_states = self.resnets[0](hidden_states, temb)

    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        
        hidden_states = attn(hidden_states)
        hidden_states, bottleneck = forward_ResnetBlock2D(resnet,hidden_states, temb, inject_into_botleneck = inject_into_botleneck)

    return hidden_states, bottleneck

    
@dataclass
class UNetOutput:
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    out: torch.FloatTensor
    h: torch.FloatTensor

class UNet:
    def __init__(self, model: UNet2DModel, h_space = None):
        assert h_space in [None, "before","after", "middle"]
        self.h_space = h_space
        self.model = model
        self.device = model.device
        
        #self.sanity_check()
        
        
    def sanity_check(self):
        assert self.model.config.center_input_sample == False
        assert self.model.config.time_embedding_type == 'positional'
        
    def time_embedding(self, timestep, batch_dim):
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(batch_dim, dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.model.time_proj(timesteps)
        emb = self.model.time_embedding(t_emb)
        return emb 
    
    
    def forward(self, sample, timestep, delta_h = None):
        # Modified from From: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d.py
     
        # 1. Positional embedding
        emb = self.time_embedding( timestep, batch_dim = sample.shape[0])

        # 2. pre-process
        skip_sample = sample        
        sample = self.model.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.h_space == "before":
            bottleneck = sample.clone()
            if not delta_h is None:
                sample = bottleneck + delta_h
        
        if self.h_space == "middle":
            sample, bottleneck = mid_block_forward(self.model.mid_block, sample, temb=emb, 
                                    encoder_states=None, 
                                    inject_into_botleneck = delta_h)

        else:
            sample = self.model.mid_block(sample, emb)


        if self.h_space == "after":
            bottleneck = sample.clone()
            if not delta_h is None:
                sample = bottleneck + delta_h

        # 5. up
        skip_sample = None
        for upsample_block in self.model.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process make sure hidden states is in float32 when running in half-precision
        sample = self.model.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample
 
        return UNetOutput(out = sample, h = None if self.h_space == None else bottleneck)

    def sample(self, num_samples = 1, seed=None):
        """
        Samples random noise in the dimensions of the Unet
        """
        if seed is None: seed = torch.randint(int(1e6), (1,))
        return torch.randn(num_samples, 
                            self.model.in_channels, 
                            self.model.sample_size,
                            self.model.sample_size,
                            generator=torch.manual_seed(seed)
                            ).to(self.device)



class ConditionalUnet(UNet):
    def __init__(self, model,  h_space = "after"):
        assert h_space in [None, "before","after" ] #"middle"
        self.h_space = h_space
        self.model = model
        self.device = model.device
    # def forward(self,sample,timestep, prompt=""):
    #     return ss
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        delta_h: Optional[torch.Tensor] = None, ## hspace activation
        ):
        default_overall_up_factor = 2**self.model.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
            
        # 0. center input if necessary
        if self.model.config.center_input_sample:
            sample = 2 * sample - 1.0
            print("[WARNING], self.config.center_input_sample")
      
        emb = self.time_embedding(timestep, batch_dim = sample.shape[0])


        sample = self.model.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    # attention_mask=attention_mask,
                    # cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if self.h_space == "before":
            bottleneck = sample.clone()
            if not delta_h is None:
                sample = bottleneck + delta_h
        # 4. mid
        sample = self.model.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            # attention_mask=attention_mask,
            # cross_attention_kwargs=cross_attention_kwargs,
        )
        if self.h_space == "after":
            bottleneck = sample.clone()
            if not delta_h is None:
                sample = bottleneck + delta_h
        # 5. up
        for i, upsample_block in enumerate(self.model.up_blocks):
            is_final_block = i == len(self.model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    # cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    # attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.model.conv_norm_out(sample)
        sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample)

        return UNetOutput(out=sample, h= None if self.h_space == None else bottleneck)

class Diffusion:
    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler, diffusers_default_scheduler = False):
        
        self.unet = unet
        self.device = self.unet.device
        
        self.scheduler = scheduler
        self.diffusers_default_scheduler = diffusers_default_scheduler
        self.sanity_check()
    
        self.update_inference_steps(num_inference_steps = 50)
    

    def update_inference_steps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps = num_inference_steps)
        self.timesteps = self.scheduler.timesteps.to(self.device)
       
        self.t_to_idx = {int(v):k for k,v in enumerate(self.timesteps)}
      
        self.h_shape  = self.get_h_shape()
        self.variance_noise_shape = (
            self.num_inference_steps,
            self.unet.model.in_channels, 
            self.unet.model.sample_size,
            self.unet.model.sample_size)

    def get_h_shape(self):
        """
        Return the shape fo the h tensors
        """
        xT = self.unet.sample()
        with torch.no_grad():   
            out = self.unet.forward(xT, timestep = self.timesteps[-1])
        return (self.num_inference_steps,) + tuple(out.h.shape[1:])


    def sanity_check(self):
        if self.scheduler.clip_sample:
            print("[Warning] Scheduler assumes clipping, setting to false")
            self.scheduler.clip_sample = False
            self.scheduler.config.clip_sample = False


    def get_variance(self, timestep): #, prev_timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    # def get_variance(self, t):
    #     # DDPM papger eq 7 
    #     alpha_bar = self.scheduler.alphas_cumprod
    #     betas = self.scheduler.betas
    #     alpha_bar_t = alpha_bar[t]
    #     alpha_bar_t_prev = alpha_bar[t - 1] if t > 0 else self.scheduler.one
    #     variance_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * betas[t]
    #     return variance_t
    
    def sample_variance_noise(self, seed=None):
        """
        Samples variance noise
        """
        if seed is None: seed = torch.randint(int(1e6), (1,))
        return torch.randn( self.variance_noise_shape,
                            generator=torch.manual_seed(seed)
                            ).to(self.device)

    def reverse_step(self, model_output, timestep, sample, eta = 0, asyrp = None, variance_noise=None):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep) #, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = model_output if asyrp is None else asyrp   
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.device)
            sigma_z =  eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample

    # Equivalent to https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
    # def reverse_step(self, model_output, timestep, sample):
    #     prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    #     alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     beta_prod_t = 1 - alpha_prod_t
    #     pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    #     pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    #     prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    #     return prev_sample

    def reverse_process(self, xT, 
                        etas = 0,
                        prog_bar = False,
                        zs = None,
                        delta_hs = None,
                        asyrp = False):
        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas]*self.num_inference_steps
        assert len(etas) == self.num_inference_steps

        xt = xT
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(self.timesteps) if prog_bar else self.timesteps 
        for t in op:
            idx = self.t_to_idx[int(t)]        
            delta_h = None if delta_hs is None else delta_hs[idx][None]
           
            with torch.no_grad():
                out = self.unet.forward(xt, timestep =  t, delta_h = delta_h)
            hs[idx] = out.h.squeeze()

            # Support for asyrp
            # ++++++++++++++++++++++++++++++++++++++++ 
            if asyrp and not delta_hs is None:
                with torch.no_grad():
                    out_asyrp = self.unet.forward(xt, timestep =  t)
                residual_d = out_asyrp.out
            else: 
                residual_d = None
            # ----------------------------------------------------
            z = zs[idx] if not zs is None else None
            
            # 2. compute less noisy image and set x_t -> x_t-1  
            xt = self.reverse_step(out.out, t, xt, asyrp = residual_d, eta = etas[idx], variance_noise = z)         
        return xt, hs, zs


    def add_noise(self, original_samples, noise, timesteps):
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        #timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.scheduler.alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


    def forward_step(self, model_output, timestep, sample):
        next_timestep = min(self.scheduler.config.num_train_timesteps - 2,
                            timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        next_sample = self.scheduler.add_noise(pred_original_sample,
                                     model_output,
                                     torch.LongTensor([next_timestep]))
        return next_sample


    # def forward_step(self, model_output, timestep, sample):
    #     # https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
    #     timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
    #     alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
    #     alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
    #     beta_prod_t = 1 - alpha_prod_t
    #     next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    #     next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    #     next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    #     return next_sample

    # def forward(self, x0, prog_bar = False):
    #     xt = x0
    #     zs = None 
    #     hs = torch.zeros(self.h_shape).to(self.device)
    #     op = tqdm(reversed(self.timesteps)) if prog_bar else reversed(self.timesteps)
    #     for t in op:
    #         idx = self.t_to_idx[int(t)]
    #         # 1. predict noise residual
    #         with torch.no_grad():
    #             out = self.unet.forward(xt, timestep =  t)
    #         hs[idx] = out.h.squeeze()
    #         # 2. compute more noisy image and set x_t -> x_t+1
    #         xt = self.forward_step(out.out, t, xt)
    #     return xt, hs, zs

    def forward(self, x0, 
                etas = None, 
                method_from = "x0",    
                prog_bar = False):

        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas]*self.num_inference_steps
            xts = self.sample_xts_from_x0(x0, method_from = method_from)
            alpha_bar = self.scheduler.alphas_cumprod
            zs = torch.zeros_like(self.sample_variance_noise())
           

        xt = x0
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(reversed(self.timesteps)) if prog_bar else reversed(self.timesteps)

        for t in op:
    
            idx = self.t_to_idx[int(t)]
            # 1. predict noise residual
            if not eta_is_zero:
                xt = xts[idx][None]
                       
            with torch.no_grad():
                out = self.unet.forward(xt, timestep =  t)
            hs[idx] = out.h.squeeze()

            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(out.out, t, xt)

            else: 
                xtm1 =  xts[idx+1][None]
                # pred of x0
                pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * out.out ) / alpha_bar[t] ** 0.5
                
                # direction to xt
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                
                variance = self.get_variance(t)
                pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * out.out   

                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
               
                z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
                zs[idx] = z
        if not zs is None: 
            zs[-1] = torch.zeros_like(zs[-1]) 

        return xt, hs, zs


    def sample_xts_from_x0(self, x0, method_from = "x0"):
        """
        Samples from P(x_1:T|x_0)
        """
        assert method_from in ["x0", "x_prev", "dpm"]
        # torch.manual_seed(43256465436)

        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
        alphas = self.scheduler.alphas
        betas = 1 - alphas

        if method_from == "x0": 
            xts = torch.zeros(self.variance_noise_shape).to(x0.device)
            for t in reversed(self.timesteps):
                idx = self.t_to_idx[int(t)]
                xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
            xts = torch.cat([xts, x0],dim = 0)

        if method_from == "x_prev": 
            xts = torch.zeros(self.variance_noise_shape).to(x0.device)
            x_next = x0.clone()

            for t in reversed(self.timesteps):   
                noise = torch.randn_like(x0) 
                idx = self.t_to_idx[int(t)]
                xt = ((1 - betas[t]) ** 0.5) * x_next + noise * (betas[t] ** 0.5)
                x_next = xt
                xts[idx] = xt
            
            xts = torch.cat([xts, x0],dim = 0)

        if method_from == "dpm":
            xts = torch.zeros(self.variance_noise_shape).to(x0.device)
            x0.clone()
            t_final = self.timesteps[0]
            xT = x0 * (alpha_bar[t_final] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t_final]
            xt = xT.clone()
            for t in self.timesteps:
                idx = self.t_to_idx[int(t)]
                xtm1 = self.sample_xtm1_from_xt_x0(xt,x0,t)
                xt = xtm1
                xts[idx] = xt
            xts = torch.cat([xts, x0],dim = 0)

        return xts


    def mu_tilde(self, xt,x0, timestep):
        "mu_tilde(x_t, x_0) DDPM paper eq. 7"
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_t = self.scheduler.alphas[timestep]
        beta_t = 1 - alpha_t 
        alpha_bar = self.scheduler.alphas_cumprod[timestep]
        return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 +  ((alpha_t**0.5 *(1-alpha_prod_t_prev)) / (1- alpha_bar))*xt

    def sample_xtm1_from_xt_x0(self, xt, x0, t):
        "DDPM paper equation 6"
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_t = self.scheduler.alphas[t]
        beta_t = 1 - alpha_t 
        alpha_bar = self.scheduler.alphas_cumprod[t]
        beta_tilde_t = ((1-alpha_prod_t_prev) / (1-alpha_bar)) * beta_t
        return  self.mu_tilde(xt,x0, t)  + beta_tilde_t**0.5 * torch.randn_like(x0)


class StableDiffusion(Diffusion):

    def __init__(self, unet, scheduler, tokenizer, text_encoder):
        self.unet = unet

        self.device = self.unet.device
        self.scheduler = scheduler
     
        self.tokenizer_id = "openai/clip-vit-base-patch32"
        self.tokenizer = tokenizer
        self.text_encoder =  text_encoder
        # self.tokenizer =  CLIPTokenizer.from_pretrained(self.tokenizer_id) #CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.text_encoder = CLIPTextModel.from_pretrained(self.tokenizer_id).to(self.device)
        self.uncond_embedding = self.encode_text("")
        self.update_inference_steps(num_inference_steps = 50)

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def get_h_shape(self):
        """
        Return the shape fo the h tensors
        """
        xT = self.unet.sample()
        text_embeddings = self.encode_text("")
        with torch.no_grad():   
            out = self.unet.forward(xT, timestep = self.timesteps[-1], encoder_hidden_states = self.uncond_embedding)
        return (self.num_inference_steps,) + tuple(out.h.shape[1:])
    

    def update_inference_steps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps = num_inference_steps)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.t_to_idx = {int(v):k for k,v in enumerate(self.timesteps)}
        
        self.h_shape  = self.get_h_shape()
  
        self.variance_noise_shape = (
            self.num_inference_steps,
            self.unet.model.in_channels, 
            self.unet.model.sample_size,
            self.unet.model.sample_size)

    def reverse_process(self, xT, 
                        etas = 0,
                        prompt = "",
                        cfg_scale = 7.5,
                        prog_bar = False,
                        zs = None,
                        delta_hs = None,
                        asyrp = False):
        

        text_embeddings = self.encode_text(prompt)

        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas]*self.num_inference_steps
        assert len(etas) == self.num_inference_steps

        xt = xT
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(self.timesteps) if prog_bar else self.timesteps 
        for t in op:
            idx = self.t_to_idx[int(t)]        
            delta_h = None if delta_hs is None else delta_hs[idx][None]
           

            ## Unconditional embedding
            with torch.no_grad():
                uncond_out = self.unet.forward(xt, timestep =  t, 
                                               encoder_hidden_states = self.uncond_embedding, delta_h = delta_h)
            hs[idx] = uncond_out.h.squeeze()

             ## Conditional embedding  
            if prompt:  
                with torch.no_grad():
                    cond_out = self.unet.forward(xt, timestep =  t, 
                                                 encoder_hidden_states = text_embeddings, delta_h = delta_h)
               
            # Support for asyrp
            # ++++++++++++++++++++++++++++++++++++++++ 
            if asyrp and not delta_hs is None:
                with torch.no_grad():
                    out_asyrp = self.unet.forward(xt, timestep =  t, encoder_hidden_states = text_embeddings)
                residual_d = out_asyrp.out
            else: 
                residual_d = None
            # ----------------------------------------------------
            z = zs[idx] if not zs is None else None
            
            if prompt:
                ## classifier free guidance
                noise_pred = uncond_out.out + cfg_scale * (cond_out.out - uncond_out.out)
            else: 
                noise_pred = uncond_out.out
            # 2. compute less noisy image and set x_t -> x_t-1  
            xt = self.reverse_step(noise_pred, t, xt, asyrp = residual_d, eta = etas[idx], variance_noise = z)         
        return xt, hs, zs

    def forward(self, x0, 
                etas = None, 
                method_from = "x0",    
                prog_bar = False):

        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas]*self.num_inference_steps
            xts = self.sample_xts_from_x0(x0, method_from = method_from)
            alpha_bar = self.scheduler.alphas_cumprod
            zs = torch.zeros_like(self.sample_variance_noise())
           

        xt = x0
        hs = torch.zeros(self.h_shape).to(self.device)
        op = tqdm(reversed(self.timesteps)) if prog_bar else reversed(self.timesteps)

        for t in op:
    
            idx = self.t_to_idx[int(t)]
            # 1. predict noise residual
            if not eta_is_zero:
                xt = xts[idx][None]
                       
            with torch.no_grad():
                out = self.unet.forward(xt, timestep =  t, encoder_hidden_states = self.uncond_embedding)
            hs[idx] = out.h.squeeze()

            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(out.out, t, xt)

            else: 
                xtm1 =  xts[idx+1][None]
                # pred of x0
                pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * out.out ) / alpha_bar[t] ** 0.5
                
                # direction to xt
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                
                variance = self.get_variance(t)
                pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * out.out   

                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
               
                z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
                zs[idx] = z
        if not zs is None:
            zs[-1] = torch.zeros_like(zs[-1]) 

        return xt, hs, zs

class Interpolations:
        
    def interpolation(self, z1, z2,
                        numsteps = 5,
                        t_min = 0, t_max = 1,
                        method = "lerp"):
        return torch.cat([self.interp(z1,z2,t, method = method) for t in torch.linspace(t_min,t_max,numsteps)])   
    
    def interp(self, z1, z2, t, method = "lerp"):
        if method == "lerp":
            return self.lerp(z1,z2,t)
        elif method == "slerp":
            return self.slerp(z1,z2,t)
        elif method == "sqlerp":
            return self.sqlerp(z1,z2,t)
        else:
            raise NotImplementedError("only lerp and slerp implemented")
            
    def slerp(self, v0, v1, t, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()
            t = t.cpu().numpy()
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)
        return v2
    
    def lerp(self, z1,z2,t):
        return z1*(1-t) + z2*t
    def sqlerp(self, z1,z2,t):
        return z1*(1-t)**0.5 + z2*t**0.5


class Q:
    """
    Representation state class
    """
    def __init__(self, x0 = None, xT = None, w0 = None, wT = None, hs = None, zs = None, etas = None, delta_hs = None, seed = None, asyrp = None, prompt = "", cfg_scale = None, label = None):    
        self.x0 = x0 # clean img
        self.xT = xT # noise img            
        self.w0 = w0 # clean latent (in case is_vq=True)
        self.wT = wT # noise latent
        
        self.zs = zs # Variance noise added (pre-sampled)
        self.hs = hs # h-space representation 
        
        # Modyfiers
        self.etas = etas #eta schedule
        self.delta_hs = delta_hs # delta hs to be injected during decoding
        self.seed = seed # seed for random number generators
        self.asyrp = asyrp # if delta hs injection is asymetrical 
        # Prompts for Stable Diffusion support
        self.prompt = prompt
        self.cfg_scale = cfg_scale # classifier free guidance scale
        self.label = label #optional label
    def copy(self):
        # return  Q(**self.__dict__.copy())
        return  Q(**copy.deepcopy(self.__dict__))
        

    def set_delta_hs(self, delta_hs):
        self.delta_hs = delta_hs    
        return self

    def to_string(self):
        string = f"seed={self.seed}-etas={self.etas}"
        if not self.label is None: 
            string += f"label={self.label}"
        string += self.prompt
        return string
    def __add__(self, other):
        return Q(delta_hs=self.delta_hs + other.delta_hs)
    def __sub__(self, other):
        return Q(delta_hs=self.delta_hs - other.delta_hs)
        
class SemanticDiffusion(Interpolations):
    def __init__(self, unet, scheduler,
                 vqvae = None, model_id = None, num_inference_steps = 25, diffusers_default_scheduler = False, resize_to = 256):

        self.vqvae = vqvae
        self.is_vq = False if self.vqvae is None else True
        self.diff = Diffusion(scheduler=scheduler, unet = unet, diffusers_default_scheduler = diffusers_default_scheduler)
        self.device = unet.device
        self.model_id = model_id
        self.is_conditional = False
        self.h_space = self.diff.unet.h_space
        self.resize_to = resize_to
        if self.is_vq:
            self.img_size = 256
        else:
            self.img_size = self.diff.unet.model.sample_size
        self.set_inference_steps(num_inference_steps = num_inference_steps)


    def get_eta_schedule(self, fraction = .2, 
                                eta_scale = 1, 
                                where = "end"):
        T = self.num_inference_steps
        nr_on = int(T*fraction)
        nr_off = T - nr_on
        if where == "end":
            etas = torch.tensor([0]*nr_off+[1]*nr_on)*eta_scale
        elif where == "end":
            etas = torch.tensor([1]*nr_on+[0]*nr_off)*eta_scale
        else:
            raise NotImplementedError
        return etas
        
    def load_model():
        pass

    def set_inference_steps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        self.diff.update_inference_steps(num_inference_steps)
        self.model_label = self.model_id.replace("/","-")+f"steps{self.num_inference_steps}-hspace-{self.h_space}"

    def encode(self, q, method_from = "x0", **kwargs):
        with autocast("cuda"), inference_mode():
            if not self.vqvae is None:
                q.w0 = self.vqvae.encode(q.x0).latents
                q.wT, q.hs, q.zs = self.diff.forward(q.w0, etas = q.etas,method_from = method_from, **kwargs)
            else:
                q.xT, q.hs, q.zs = self.diff.forward(q.x0, etas = q.etas,method_from = method_from,  **kwargs)
        return q

            
    def decode(self, q, **kwargs):
        if self.is_vq : #not self.vqvae is None:
            q.w0, q.hs, q.zs= self.diff.reverse_process(q.wT,
                                                   zs = q.zs,
                                                   etas= q.etas,
                                                   delta_hs = q.delta_hs, 
                                                   asyrp=q.asyrp,
                                                   **kwargs)
            with autocast("cuda"), inference_mode():
                q.x0 = self.vqvae.decode(q.w0).sample
        else:

            q.x0, q.hs, q.zs = self.diff.reverse_process(q.xT,
                                                   zs = q.zs,
                                                   etas = q.etas,
                                                   delta_hs = q.delta_hs,
                                                   asyrp=q.asyrp, 
                                                   **kwargs)
        return q 

    def sample(self, decode = True, 
               seed=None,
               prompt = "",
               variance_seed = None,
               etas=None, **kwargs):
        """
        Samples random noise in the dimensions of the Unet
        """
        if seed is None: seed = torch.randint(int(1e6), (1,))
        q = Q(seed = seed, etas = etas, prompt = prompt)
        sample = self.diff.unet.sample(seed = seed)
        if not self.vqvae is None: q.wT = sample
        else: q.xT = sample 
        if not etas is None: 
            if variance_seed is None:
                variance_seed = seed + 1 
            ## Very important the the first zt is not equal to xT 
            ## if seed xT and seed zs is equal horrible stuff happens
            q.zs = self.sample_variance_noise(seed = variance_seed) 
          
        if decode: 
            q = self.decode(q,**kwargs)
        return q

    def sample_seeds(self, etas = None, num_imgs = 25, imsize = None, plot_seed_nr = True, rows = 5, cols = 5 ):
        qs = [self.sample(etas = etas) for _ in tqdm(range(num_imgs))]
        imgs = [q.x0 for q in qs]
        labels = [str(int(q.seed)) for q in qs] if plot_seed_nr else None 
        return image_grid(imgs, titles=labels,size=imsize, rows = rows, cols = cols)

    def sample_variance_noise(self, seed=None):
        return self.diff.sample_variance_noise(seed = seed)
    
    def apply_direction(self, q, n, scale = 1, space = "hspace"): 
        q_edit = q.copy()   
        if space == "noise":
            if self.is_vq:
                q_edit.wT = q_edit.wT + scale*n.wT.to(self.device) 
            else:
                q_edit.xT = q_edit.xT + scale*n.xT.to(self.device)
            if not q_edit.etas is None:
                q_edit.zs = q_edit.zs + scale*n.zs.to(self.device) if not n.zs is None else q_edit.zs 
        elif space == "hspace":
            q_edit.delta_hs = scale*n.delta_hs.to(self.device)    
        q_edit = self.decode(q_edit) 
        return q_edit

    def interpolate_direction(self, 
                              q, n, 
                    space = "hspace", 
                    t1 = 0, t2 = 1, 
                    numsteps = 5,
                    vertical = False,
                    plot_strength = True,
                    ):
        qs = []
        interval = torch.linspace(t1,t2,numsteps)
        for t in interval:
            q_edit= self.apply_direction( q, n, scale = t, space = space)
            qs.append(q_edit)
    
        imgs = torch.cat([q.x0 for q in qs])
        interval = [str(i.item()) for i in interval]
        if plot_strength == False:
            interval = None

        if vertical:
            plot = image_grid(imgs, titles = interval, cols = 1, rows = len(imgs), size = self.resize_to)
        else:
            plot = image_grid(imgs, titles = interval, size = self.resize_to)
        
        return plot

    def interpolate(self, q1, q2, 
                    space = "pixel", 
                    method = "lerp",
                    t1 = 0, t2 = 1, numsteps = 5,
                    to_img = False):
        qs = []
        
        for t in tqdm(torch.linspace(t1,t2,numsteps)):
    
            q = q1.copy()
            if space == "pixel":
                q.x0 = self.interp(q1.x0, q2.x0, t, method=method)         

            elif space == "noise":
                if self.is_vq:
                    q.wT = self.interp(q1.wT, q2.wT, t, method=method)
                else:
                    q.xT = self.interp(q1.xT, q2.xT, t, method=method)
                if not q.etas is None:
                    q.zs = self.interp(q1.zs, q2.zs, t, method=method)
                q = self.decode(q)  

            elif space == "hspace":
                if self.is_vq: q.wT = q1.wT
                else: q.xT = q1.xT
                q.delta_hs = self.interp(torch.zeros_like(q2.delta_hs), 
                                         q2.delta_hs, t, method=method)
                q = self.decode(q) 
            elif space == "vq-denoisedspace":
                assert self.is_vq
                raise NotImplementedError
                # q.wT = self.interp(q1.wT, q2.wT, t, method=method
            else: 
                raise NotImplementedError
            qs.append(q)
        if to_img: qs = torch.cat([q.x0 for q in qs])   
        return qs

    def load_real_image(self, folder = "data/real_images/", idx = 0):

            path = glob(folder + "*")[idx]
            img = Image.open(path).resize((self.img_size, 
                                        self.img_size))

            img = pil_to_tensor(img).to(self.device)

            if img.shape[1]== 4:
                img = img[:,:3,:,:]
            return img

    def img_path_to_tensor(self, path):
        img = Image.open(path).resize((self.img_size, 
                                       self.img_size))
        return pil_to_tensor(img).to(self.device)

    def show(self, q):
        if not type(q) == list:
            q = [q]
        imgs = [q_.x0 for q_ in q]
        
        return image_grid(imgs, size = self.resize_to)
    
    def inner_product(self, a,b):
        return (a*b).sum() 
    

class StableSemanticDiffusion(SemanticDiffusion):

    def __init__(self, unet, scheduler, vae, tokenizer, text_encoder,  model_id = None, num_inference_steps = 25  ):
        self.img_size = 512
        self.device = unet.device
        self.unet = unet
        self.vae = vae
        self.is_vq = True
        self.is_conditional = True
        self.resize_to = 256
        self.diff = StableDiffusion(unet = unet, 
                        scheduler=scheduler,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder)
        self.model_id = model_id 
        self.h_space = self.diff.unet.h_space
        self.set_inference_steps(num_inference_steps = num_inference_steps)
        
    def decode(self, q, **kwargs):
        q.w0, q.hs, q.zs= self.diff.reverse_process(q.wT,
                                                   zs = q.zs,
                                                   prompt=q.prompt,
                                                   etas= q.etas,
                                                   delta_hs = q.delta_hs, 
                                                   asyrp=q.asyrp,
                                                   **kwargs)
        with autocast("cuda"), inference_mode():
            q.x0 = self.vae.decode(1 / 0.18215 * q.w0).sample
        
        return q 
        
    def encode(self, q, method_from = "x0", **kwargs):
        with autocast("cuda"), inference_mode():
            q.w0 = (self.vae.encode(q.x0).latent_dist.mode() * 0.18215).float()
        q.wT, q.hs, q.zs = self.diff.forward(q.w0, etas = q.etas,method_from = method_from, **kwargs)
        return q

    def sample(self, decode = True, 
               seed=None,
               prompt = "",
               variance_seed = None,
               etas=None, **kwargs):
        """
        Samples random noise in the dimensions of the Unet
        """
        if seed is None: seed = torch.randint(int(1e6), (1,))
        q = Q(seed = seed, etas = etas, prompt = prompt)
        sample = self.diff.unet.sample(seed = seed)
        q.wT = sample 
        if not etas is None: 
            ## Very important the first zt is not equal to xT 
            ## if seed of xT and seed of zs is equal horrible stuff happens (but only sometimes)
            if variance_seed is None:
                variance_seed = seed + 1 

            q.zs = self.sample_variance_noise(seed = variance_seed) 
          
        if decode: 
            q = self.decode(q,**kwargs)
        return q
    

def load_model(model_id, 
                  device = "cuda",
                  h_space = "after",
                  scheduler = "ddim",
                  num_inference_steps = 25):
   
   # Support for Stable diffusion - Not used in the paper
   if model_id in ["CompVis/stable-diffusion-v1-4",
                   "stabilityai/stable-diffusion-2", 
                   "runwayml/stable-diffusion-v1-5"]:

     
      # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
      stable_diffusion = StableDiffusionPipeline.from_pretrained(model_id).to(device) #use_auth_token=True
      if model_id == "CompVis/stable-diffusion-v1-4":
         scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
      else:
         scheduler = stable_diffusion.scheduler

      return StableSemanticDiffusion(
         unet=ConditionalUnet(stable_diffusion.unet),
         scheduler=scheduler,#DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
         vae = stable_diffusion.vae,
         tokenizer = stable_diffusion.tokenizer,
         text_encoder = stable_diffusion.text_encoder,
         model_id = model_id,
         num_inference_steps=num_inference_steps
      )
      
   if "pixel" == model_id: 
      model_id = "google/ddpm-ema-celebahq-256"
   if "ldm" == model_id: 
      model_id = "CompVis/ldm-celebahq-256"

   vqvae = None
   if "ldm" in model_id:
      vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae").to(device)
   
   try:      
      model = UNet2DModel.from_pretrained(model_id).to(device)
      scheduler = DDIMScheduler.from_pretrained(model_id)
   except:
      if "stable-diffusion" in model_id:
         model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
         vqvae = VQModel.from_pretrained(model_id, subfolder="vae")
      else:
        try:
            model = UNet2DModel.from_pretrained(model_id, subfolder="unet").to(device)
        except:
            model = UNet2DModel.from_pretrained(model_id).to(device)
      scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
 
   unet = UNet(model, h_space=h_space)
   sd = SemanticDiffusion(unet, scheduler, vqvae = vqvae, 
                          model_id = model_id, 
                          num_inference_steps = num_inference_steps)
   return sd