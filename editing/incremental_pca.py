import os 
import torch

import sys
sys.path.append("..")  # Add the parent directory to the path

from utils import image_grid
from semanticdiffusion import load_model, Q
from utils import tensor_to_pil

from sklearn.decomposition import IncrementalPCA

from tqdm.auto import tqdm

from dataclasses import dataclass
from editing.pca import make_sure_vecs_point_in_same_direction
import pyrallis

def do_pca(sd, cfg):


    os.makedirs(cfg.cache_dir, exist_ok=True)
    out_path = cfg.cache_dir + f"pca-{sd.model_label}-samples-{cfg.num_samples}-etas-{cfg.etas}-n_pcs{cfg.n_components}.pt"
    if os.path.exists(out_path) and not cfg.force_rerun:
        print("Loading:", out_path)
        return torch.load(out_path)
    print(out_path)
    print("Calculating PCs")
    torch.manual_seed(42)
    IPCA = IncrementalPCA(n_components=cfg.n_components)
    IPCAs = [IncrementalPCA(n_components=cfg.n_components)  for i in range(cfg.num_inference_steps)]
    num_batches = cfg.num_samples // cfg.batch_size
    for _ in tqdm(range(num_batches)):
        hhs =  torch.cat([sd.sample(etas = cfg.etas).hs.detach().cpu()[None] for _ in tqdm(range(cfg.batch_size))])
        hhs_shape = hhs.shape
        hhs_flat_all = hhs.flatten(1)
        IPCA.partial_fit(hhs_flat_all)

        hhs_flat_idv = hhs.flatten(2)

        for i, ihhs in enumerate(hhs_flat_idv.permute(1,0,2)):
            IPCAs[i].partial_fit(ihhs)
        
        del hhs
        del hhs_flat_idv

    Us_all = torch.tensor(IPCA.components_.reshape((cfg.n_components,) + hhs_shape[1:]))
    svals_all = IPCA.singular_values_
    PCs_all = torch.cat([s*u[None] for u,s in zip(Us_all, svals_all)], axis = 0)

    Us_indv = torch.cat([torch.tensor(ipca.components_).reshape((cfg.n_components,) + hhs_shape[2:]).unsqueeze(1) for ipca in IPCAs],axis = 1)
    svals_indv = torch.cat([torch.tensor(ipca.singular_values_)[None] for ipca in IPCAs],axis = 0).T
    
    PCs_indv = torch.cat([torch.cat([s*u[None] for u,s in zip(U,sval)],axis = 0)[None] 
                    for U, sval in zip(Us_indv,svals_indv)], axis = 0)
    PCs_indv = make_sure_vecs_point_in_same_direction(PCs_indv)

    result = {"all": [Us_all, svals_all, PCs_all],
              "indv": [Us_indv, svals_indv, PCs_indv]}
    
    torch.save(result, out_path) 
    return result

def plot_pc(sd, q, PCs, strength = 1, nr_pcs = 5):
    imgs = []
    for pc in tqdm(PCs[:nr_pcs]):
        qs = [q.copy().set_delta_hs(pc.to(q.hs.dtype)*a) for a in torch.linspace(-strength,strength,5)]
        qs = [sd.decode(q) for q in qs]
        img = sd.show(qs)
        imgs.append(img)
    return image_grid(imgs, rows = len(imgs), cols = 1)


@dataclass
class Config:
    num_samples: int = 50000 
    batch_size: int = 1000
    n_components: int= 25
    etas: int = 1
    num_inference_steps: int = 50 
    method: str = "indv"
    force_rerun: bool = False 
    cache_dir: str = "results/pca/"
    fig_dir: str = "results/pca/figs/"
    plot_nr_pcs: int = 15
    seeds: list = (82867, 259143, 765297, 2543) 
    device: str = "cuda"



def main():

    cfg = pyrallis.parse(config_class=Config)
    sd = load_model("pixel", device = cfg.device,
                    h_space = "after",
                    num_inference_steps = cfg.num_inference_steps)


    results = do_pca(sd, cfg)
    Us, svals, PCs = results[cfg.method]
    PCs = PCs.to(sd.device)

    print("Plotting results")
    os.makedirs(cfg.fig_dir, exist_ok=True)
    fig_path = cfg.fig_dir + f"{cfg.num_samples}-{cfg.num_inference_steps}-{cfg.method}.jpg"
    if not os.path.exists(fig_path):    
        qs = [sd.sample(etas = cfg.etas, seed = s) for s in tqdm(cfg.seeds)]
        imgs = [plot_pc(sd, q, PCs, strength = 0.1, nr_pcs=cfg.plot_nr_pcs) for q in qs]
        image_grid(imgs).save(fig_path)
    print("done")




if __name__ == "__main__":
    main()