import os
import math
import torch 
from tqdm import tqdm 

from functools import partial 
from dataclasses import dataclass
from matplotlib import pyplot as plt

from utils import image_grid
from semanticdiffusion import Q


def make_sure_vecs_point_in_same_direction(V):
    V = V.clone()
    counter = 0
    for j in range(V.shape[0]): # loop over all PCs
        for i in range(V.shape[1]): # loop over inference steps
            if i > 0: # Take the first inference step as reference
                ip = (V[j,0]*V[j,i]).sum() # inner product
                if ip < 0: # if negative, change sign
                    counter +=1
                    V[j,i] = - V[j, i]
                    ip = (V[j,0]*V[j,i]).sum()
                    assert ip > 0 
    print("[INFO] SIGN Ambiguity: changed sign on", counter, "vectors")
    return V

def fix_sign_ambiguity(x):
    if torch.sign(x[0]) == -1:
        return -x
    else:
        return x


class DirectionPlotter:

    def __init__(self, sd):
        self.sd = sd
        self.device = sd.device

    def plot_direction(self, q, hhs_dirs,
                             num_pcs = 5,
                             point_same_direction = True, 
                             scale = 200,
                             alpha_font_size = 30,
                             plot_pc_number = False,
                             numsteps = 5):
        if point_same_direction:
            hhs_dirs =  make_sure_vecs_point_in_same_direction(hhs_dirs)

        imgs = []
        for i in tqdm(range(num_pcs)):
            hs = hhs_dirs[i].to(self.device)
            lerp = self.sd.interpolate_direction(q, Q(delta_hs = hs), 
                    space = "hspace", t1 = -scale, t2 = scale,
                    numsteps = numsteps)         
            if plot_pc_number:
                lerp = image_grid([lerp],titles = [f"PC #{i}"], font_size=30, top = 30)
            imgs.append(lerp)
        img = image_grid(imgs,cols = 1, rows = len(imgs),  add_margin_size=15)
      
        return img
    
    def power_direction_overview(self, q, Uts, ssvals, 
                                    global_at_idx = False,
                                    scale  = 1,
                                    point_same_direction = True,
                                    plot_every = 1,
                                    t_idx_font_size = 40,
                                    scale_fontsize = 30,
                                    apply_svals = True,
                                    numsteps = 5,
                                    svec_idx = 0,
                                    window = 1,              
                                    ):
        
        if point_same_direction:
            Uts =  make_sure_vecs_point_in_same_direction(Uts)
        svals = ssvals[svec_idx]
        imgs = []
        labels = []
        for t_idx in tqdm(range(self.sd.num_inference_steps)):
            hs = Uts[svec_idx]
            if t_idx % plot_every == 0:
                if global_at_idx:
                    hs = hs[t_idx][None].repeat(self.sd.num_inference_steps,1,1,1)
                
                hs_local = torch.cat([h[None] if abs(t_idx-idx) <= window  else torch.zeros_like(h)[None]  for idx, h in enumerate(hs)])
                if apply_svals:
                    hs_local = torch.cat([(s*h).unsqueeze(0) for s,h in zip(svals, hs_local)])
                lerp = self.sd.interpolate_direction(q, Q(delta_hs = hs_local), 
                                    space = "hspace", t1 = -scale, t2 = scale, # font_size = scale_fontsize,
                                    numsteps = numsteps, vertical = False)
                imgs.append(lerp)
                labels.append(f"t={self.sd.diff.timesteps[t_idx]}") #idx={t_idx}-
        imgs = image_grid(imgs,titles=labels, rows = len(imgs), add_margin_size=10, cols = 1, font_size=t_idx_font_size, top = t_idx_font_size )
    
        return imgs

    def get_direction(self, Uts, ssvals, 
                                    global_at_idx = False,
                                    svec_idx = 0,
                                    point_same_direction = True,
                                    t_idx = 0,
                                    apply_svals = True,
                                    window = 999,              
                                    ):

        if point_same_direction:
            Uts =  make_sure_vecs_point_in_same_direction(Uts)
            
        svals = ssvals[svec_idx]
        hs = Uts[svec_idx]
        
        if global_at_idx:
            hs = hs[t_idx][None].repeat(self.sd.num_inference_steps,1,1,1)
        
        hs_local = torch.cat([h[None] if abs(t_idx-idx) <= window  else torch.zeros_like(h)[None]  for idx, h in enumerate(hs)])
        if apply_svals:
            hs_local = torch.cat([(s*h).unsqueeze(0) for s,h in zip(svals, hs_local)])

        n = Q(delta_hs = hs_local)
       
        return n

    def plot_timedependent_svals(self, svals, nr = 5):
        n,_ = svals.shape
        if n < nr:
            nr = n
        timesteps = self.sd.diff.timesteps.cpu()
        for i in range(nr):
            plt.plot(timesteps, svals[i], label = f"sval #{i+1}")
        plt.xlim(max(timesteps)+50, min(timesteps)-50)
        plt.legend()
        plt.ylabel("Singular value")

        plt.xlabel("Timestep t")
