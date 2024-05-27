import os
import math
import torch 
from tqdm import tqdm 

from functools import partial 
from dataclasses import dataclass
from matplotlib import pyplot as plt

from utils import image_grid
from semanticdiffusion import Q

from editing.direction_plotter import fix_sign_ambiguity, DirectionPlotter


class PowerItteration:
    
    def power_iteration_method(self, 
                                f, h,tol = 1e-7, 
                                max_iters = 50,
                                prog_bar = False,
                                v_perp = None,
                                v_init = None):
        """
                ## steps for power itteration method 
                # (1) sps matrix A  = J.T @ J  
                # (2) v_hat = Av
                # (3) v = v_hat / norm(vhat)
                # (4) Av = J.T @ J v 
                # (5) d/dh < y, h> = Jv = q
                # define h = z + aq (where a is some scaler and z is the diff btw h and a*q)
                # now d/da f(z + aq) = d/da f(h)  = J.T @ q
        """
        if not v_perp is None: 
            v_perp = v_perp.to(h.device)
            part1 = v_perp @ (v_perp.T @ v_perp).inverse()
            part2 = v_perp.T

        h.requires_grad = True 

        ## Step 1 (inialize v)
        y = f(h).view(-1)  # flatten 

        if v_init is None:
            v = torch.randn_like(y) 
            v = v / torch.sqrt(v@v)
        else: 
            v =  v_init.to(h.device)
        tol_counter = 0
        eigval_prev = 999
        op = tqdm(range(max_iters)) if prog_bar else range(max_iters)
        
        
        for iteration in op:   
            ## d/dh < y, h> = Jv = q 
            Jv = torch.autograd.grad( y @ v, h, retain_graph=True)[0].detach().clone()

            ## Now calcualte J.T @ q (where q is an arbitrary vector)
            a = torch.ones(1, requires_grad=True, device=h.device)
            z = h - a * Jv
            z = z.detach().clone() 
            Jv = Jv.contiguous()

            #Calculate  J.T @ J v 
            f_tilde = lambda a_:  f(z + a_* Jv )
            JtJv = torch.autograd.functional.jacobian(f_tilde, a, vectorize=True,  strategy='forward-mode').detach() #create_graph=False, strict=False, vectorize=True, strategy='forward-mode'

            v_hat = JtJv.flatten()
            # Projection on the orthogonal complement
            if not v_perp is None: 
                v_hat_projected = part1 @ (part2 @ v_hat.detach())
                v_hat = v_hat - v_hat_projected
            
            eigval = (v_hat*v_hat).sum() ** 0.5
            sval = eigval ** 0.5
            v = v_hat / eigval 
            
            err = abs(eigval-eigval_prev)
            eigval_prev = eigval
          
            if err < tol:
                tol_counter += 1
                if tol_counter > 10:
                    break
    
        if iteration + 1 == max_iters:
            print("[WARNING], max iters reached")
            print("final err", err)

        return v, sval, Jv/sval
    
    
    def sequential_power_iteration(self, 
                                        f, h, v_init = None, 
                                        tol = 1e-6, k = 3, 
                                        max_iters = 150, etas = None):
        us = []
        ss = []
        vs = []
        v_perp = None
        for i in range(k):

            v,s,u = self.power_iteration_method(f, h, tol = tol, 
                                                        prog_bar=False, 
                                                        max_iters = max_iters,
                                                        v_perp = v_perp,
                                                        v_init = v_init if v_init is None else v_init[:,i-1] 
                                                        )
            vs.append(v)
            ss.append(s)
            us.append(u)
            v_perp = torch.hstack([v.unsqueeze(1) for v in vs])
            v_init = v_perp
        if len(us[0].shape) == 1:
            us = torch.cat([u.unsqueeze(0) for u in us])
        else: 
            us = torch.cat(us)
        if len(vs[0].shape) == 1:
            vs = torch.cat([v.unsqueeze(0) for v in vs])
        else: 
            vs = torch.cat(vs)
        ss = torch.cat([s.unsqueeze(0) for s in ss])  
        return vs,ss,us

    def test(self):
        def f(v):
            return torch.sin(A@v)
        A = torch.randn((10,100)).double()
        v0 = torch.randn(100).double()

        tol = 1e-6
        k = 3
        power = PowerItteration()
        vs, ss, us = power.sequential_power_iteration(f,v0, max_iters = 1000, tol=1e-18,k = k)

        Jgt = torch.autograd.functional.jacobian(f,v0)
        Ugt,Sgt,Vtgt = torch.linalg.svd(Jgt)
        Vtgt = Vtgt[:k]
        Ugt = Ugt.T[:k]
        Vtgt = torch.cat([fix_sign_ambiguity(x)[None] for x in Vtgt]) 
        Ugt = torch.cat([fix_sign_ambiguity(x)[None] for x in Ugt]) 
        vs = torch.cat([fix_sign_ambiguity(x)[None] for x in vs]) 
        us = torch.cat([fix_sign_ambiguity(x)[None] for x in us]) 
        ss, Sgt[:k]
        us.shape, Vtgt.shape
        assert abs(Sgt[:k] - ss).sum() < tol 
        assert (abs(vs-Ugt) < tol).all()

        assert (abs(us-Vtgt) < tol).all()
        print("Test passed")




class PowerItterationMethod:
    
    def __init__(self, sd, results_folder = "results/poweriter/"):
        torch.manual_seed(42)
        self.sd = sd
        self.device = sd.device
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

        self.dp = DirectionPlotter(sd)  

        self.sd.diff.unet.model.eval()
        for p in self.sd.diff.unet.model.parameters():
            p.requires_grad = False
    
    def load_power_dirs(self, q, 
                            num_eigvecs = 3, 
                            tol = 1e-5, 
                            mask = None,
                            max_iters = 50, 
                            prog_bar = True, 
                            force_rerun = False):

        # Unique path and load
        out_path = self.results_folder + self.sd.model_label + q.to_string() + f"num={num_eigvecs}-tol={tol}.pt"
        if os.path.exists(out_path) and not force_rerun:
            svals, svecs = torch.load(out_path)
            # print("[INFO] Loaded from", out_path)
        else: 
            svals, svecs = self.calc_power_dirs(q, prog_bar = prog_bar, max_iters = max_iters, mask=mask, tol = tol, num_eigvecs = num_eigvecs)
            torch.save([svals, svecs], out_path)
            # print("[INFO], saved to", out_path)
     
        return svals, svecs

    def calc_power_dirs(self, q,
                            mask = None,
                            num_eigvecs = 3, 
                            tol = 1e-5, prog_bar = True, 
                            max_iters = 50):


        power = PowerItteration()
    
        xt = q.xT.clone()
        xt.requires_grad = False
        op = tqdm(self.sd.diff.timesteps) if prog_bar else self.sd.timesteps 
        v_init = None 
        svals = torch.zeros((num_eigvecs, self.sd.num_inference_steps))
        svecs = torch.zeros((num_eigvecs,) + tuple(q.hs.shape))
        for t in tqdm(op):

            idx = self.sd.diff.t_to_idx[int(t)]       
            with torch.no_grad():            
                out = self.sd.diff.unet.forward(xt, timestep =  t)
            h = out.h
           
            def f_(h):
                out = self.sd.diff.unet.forward(xt, timestep =  t, delta_h = -h.detach() + h).out
                if not mask is None:
                    out = out*mask
                return out     
            
         
            V,S,U = power.sequential_power_iteration(f_,h,tol = tol,max_iters=max_iters, k = num_eigvecs)
            xt = self.sd.diff.reverse_step(out.out, t, xt, eta = q.etas) 
        
            svals[:,idx] = S      
            svecs[:,idx] = U

        return svals, svecs
    
    def make_plot(self,q,scale = 75,
                        plot_every = 5,
                        numsteps = 5,
                        point_same_direction = True,
                        t_idx_font_size = 30,
                        scale_fontsize = 30,
                        svec_title_fontsize = 60,
                        num_eigvecs = 3
        ):

        svals, svecs = self.load_power_dirs(q, num_eigvecs = num_eigvecs, 
                            tol = 5e-6, prog_bar = True, 
                            max_iters = 200,
                            multiply_by_svecs = False,
                            force_rerun = False)

        imgs = [self.dp.power_direction_overview(q, svecs, svals, 
                                                    global_at_idx = True,
                                                    scale  = scale,
                                                    plot_every = plot_every,
                                                    point_same_direction = point_same_direction,
                                                    apply_svals = True,
                                                    scale_fontsize = scale_fontsize,
                                                    t_idx_font_size = t_idx_font_size,
                                                    numsteps = numsteps,
                                                    svec_idx = i,
                                                    window = 999) for i in range(num_eigvecs)]
        
        img = image_grid(imgs, titles = [f"Singular Vector #{i}" for i in range(len(imgs))], add_margin_size=10, font_size=svec_title_fontsize, top=svec_title_fontsize)
        return img 
    

