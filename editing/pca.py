import os
import math
import torch 
from functools import partial 
from tqdm import tqdm 
from dataclasses import dataclass
from utils import image_grid

from criteria.lpips.lpips import LPIPS
from semanticdiffusion import Q


from matplotlib import pyplot as plt
from torch.nn import MaxPool2d, AvgPool2d
import itertools


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


def make_variations(sd, q, delta = 0.15, num = 16, return_n_imgs = False,force_rerun = False):
    os.makedirs("tmp/",exist_ok=True)

    out_path = "tmp/" + f"variations-pca-{sd.model_label}-samples-{num}-{q.to_string()}.pt"
    if os.path.exists(out_path) and not force_rerun:
        hhs = torch.load(out_path)
        print("[INFO] Loaded from", out_path)
        return hhs, []

    q_edit = q.copy()
    imgs = []
    hhs = []
    for i in tqdm(range(num)):
        if sd.is_vq:
            q_edit.wT = (1-delta) ** 0.5 * q.wT + delta**0.5 * torch.randn_like(q.wT) 
        else:
            q_edit.xT = (1-delta) ** 0.5 * q.xT + delta**0.5 * torch.randn_like(q.xT) 

        if not q.etas in [None, 0]:
            q_edit.zs = (1-delta) ** 0.5 * q.zs + delta**0.5 * torch.randn_like(q.zs) 
       
        q_edit = sd.decode(q_edit)
        
        #return first n images for sanity check
        if return_n_imgs and i<return_n_imgs:
            imgs.append(sd.show(q_edit))

        hs = q_edit.hs.detach().cpu()[None]
        hhs.append(hs)
    hhs = torch.cat(hhs)
    torch.save(hhs, out_path)
    print("[INFO] Saved to",out_path)
    return hhs, imgs

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
                    space = "hspace", t1 = -scale, t2 = scale,font_size = alpha_font_size,
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
                                    space = "hspace", t1 = -scale, t2 = scale, font_size = scale_fontsize,
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

# ---------------------------------------------------------------

class SVDViaKernelMatrix:

    """
    X in (n,m)
    X = V @ S^{1/2} @ U.T 
    C = X.T @ X in (m,m) = U @ S @ U.T 
    K = X @ X.T in (n,n) = V @ S @ V.T 
    so

    U.T = S^{-1/2} @ V.T @ X
    """
    def __init__(self, fix_sign = True,center = True,scale=True):
        self.fix_sign = fix_sign
        self.center = center 
        self.scale = scale


    def svd(self, X):
        n,m = X.shape
        X = X.double()
        if self.center:
            X_mean = X.mean(0)
            X = X-X_mean
        # if self.scale:
        #     X = X/(n-1)
        V,S,_ = torch.linalg.svd(X @ X.T)
        svals = S**0.5

        Ut = torch.diag(svals**-1) @ V.T @ X
    
        if self.fix_sign:
            Ut = torch.cat([fix_sign_ambiguity(x)[None] for x in Ut]) 
        # svals = (S/(n-1))**0.5
        return  Ut, svals

    def test(self, tol = 1e-8):
        A = torch.randn((100,2000)).double()
        n,m = A.shape

        if self.center:
            A = A-A.mean(0)

        Vgt, Sgt, Ugt = torch.linalg.svd(A,full_matrices=False)

 
        Ut, svals = self.svd(A)
        Ugt = torch.cat([fix_sign_ambiguity(x)[None] for x in Ugt]) 
        Ut = torch.cat([fix_sign_ambiguity(x)[None] for x in Ut]) 

        # Discard the last svec
        Ugt = Ugt[:-1] 
        Ut = Ut[:-1] 
        svals = svals[:-1]
        Sgt = Sgt[:-1]

        assert abs((svals-Sgt).sum()) < tol
        assert abs((Ut-Ugt).sum()) < tol
        print("test Passed")

class PCAMethod:
    def __init__(self, sd, runtime_seed = 42, cache = "results/pca/"):
        torch.manual_seed(runtime_seed)
        self.sd = sd
        self.device = sd.device
        self.pca = SVDViaKernelMatrix(center = True, fix_sign=True)
        self.cache = cache
        os.makedirs(self.cache, exist_ok=True)

    def sample(self, num_samples = 100, force_rerun = False, etas = None):
        out_path = self.cache + f"pca-{self.sd.model_label}-samples-{num_samples}-etas-{etas}-raw.pt"
        if os.path.exists(out_path) and not force_rerun:
            hhs = torch.load(out_path)
            print("[INFO] Loaded from", out_path)
            return hhs
        # torch.manual_seed(439873298423)
        hhs = []
        for _ in tqdm(range(num_samples)):
            hs = self.sd.sample(etas = etas).hs.detach().cpu()[None]
            hhs.append(hs)
        hhs = torch.cat(hhs)
        torch.save(hhs, out_path)
        print("[INFO] Saved to",out_path)
        return hhs
    
    def get_PCs_all(self, hhs, num_svectors = 25):
        num_samples = hhs.shape[0]
        X = hhs.view(num_samples,math.prod(self.sd.diff.h_shape)).double()
        feat_shape = hhs.shape[1:]
        Ut, svals = self.pca.svd(X)
        
        svals = svals[:num_svectors]
        Ut = Ut[:num_svectors]
        
        
        n,m = X.shape
        svals_scaled = svals / (n-1)**0.5
        Ut = Ut.reshape((num_svectors,) + feat_shape).float()

        PCs = torch.cat([(s*U).unsqueeze(0) for s,U in zip(Ut,svals_scaled)])
        
        return PCs, svals_scaled, Ut
    
    def get_PCs_indv(self, hhs, num_svectors = 25):

        N, I  = hhs.shape[:2]
        individual_feat_shape = hhs.shape[2:]
        
        # Flatten across inference steps
        # Eg feature size 512 x 8 x 8
        hhs_pr_step = (hhs[:,i].reshape(N,math.prod(individual_feat_shape)) for i in range(self.sd.num_inference_steps))
    
        # Do PCA at each inference step
        Uts, ss = [], []
        for X in tqdm(hhs_pr_step):
            Ut, svals = self.pca.svd(X)

            svals = svals[:num_svectors]
            Ut = Ut[:num_svectors]

            Uts.append(Ut)
            ss.append(svals[None])

        # Concat results
        ss = torch.cat(ss).T  
    
        ss_scaled = ss / (N-1) ** 0.5                                          
        # Uts = torch.cat([U.reshape((N,) + individual_feat_shape).unsqueeze(1) for U in Uts], dim=1).float()
        Uts = torch.cat([U.reshape((num_svectors,) + individual_feat_shape).unsqueeze(1) for U in Uts], dim=1).float()

        PCs = torch.zeros_like(Uts)
        for i,j in itertools.product(range(num_svectors),range(I)):
            PCs[i,j] = ss_scaled[i,j]*Uts[i,j]
    
        return PCs, ss_scaled, Uts

class PCAManipulator(PCAMethod):
    def __init__(self, sd, num_samples = 500, sample_etas = None):
        super().__init__(sd)
        # self.sd = sd
        hhs = self.sample(num_samples=num_samples, etas = sample_etas)
        self.PCs, self.ss, self.Uts = self.get_PCs_all(hhs)

    def apply_direction(self, q, eig_idx = 0, strength = 1, global_at_idx = False):
        hhs = self.PCs[eig_idx]
        q_edit = self.sd.apply_direction(q, Q(delta_hs=hhs), scale = strength, space = "hspace")
        return q_edit
        