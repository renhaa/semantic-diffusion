import os
import sys
import torch 
from tqdm import tqdm 

import PIL
from PIL import Image
from glob import glob

from semanticdiffusion import Q
from utils import image_grid, pil_to_tensor

class BU3DFEDirections:
    def __init__(self, sd, 
                path_to_bu3dfe = "data/bu3dfe/processed/", 
                out_folder = "tmp/bu3dfe/", 
                etas = None, 
                # max_num_samples = 100
                use_3d = False, 
                ):
        self.fs = glob(path_to_bu3dfe + "*", )
        self.out_folder = out_folder
        self.use_3d = use_3d

        os.makedirs(self.out_folder, exist_ok=True)

        self.expr_to_idx = {
           "neutral":"NE00", 'anger':"AN04", 'disgust':"DI04", 'fear':"FE04",'happiness':"HA04", 'sadness':"SA04", 'surprise':"SU04"
        }
        self.dirs = ['anger', 'disgust', 'fear','happiness', 'sadness', 'surprise']
        self.etas = etas
        self.ns = {}
        self.sd = sd

    def calc_dirs(self,  force_rerun = False):
        for d in self.dirs:
            self.calc_direction(d, force_rerun = force_rerun)

    def get_paths(self, label):
        if self.use_3d:
            paths = sorted([f for f in self.fs if self.expr_to_idx[label] in f])
        else:
            paths = sorted([f for f in self.fs if "2D" in f and self.expr_to_idx[label] in f])
        return paths

    def load_and_resize(self, f):
        img = Image.open(f)
        img = img.resize((self.sd.img_size, self.sd.img_size), PIL.Image.ANTIALIAS)
        img = pil_to_tensor(img).to(self.sd.device)
        return img
    

    def calc_direction(self, label, force_rerun = False):
        dir_path = self.out_folder + self.sd.model_label 
        if self.use_3d:
            dir_path += f"-etas{self.etas}label{label}-2D.pt"
        else:
            dir_path += f"-etas{self.etas}label{label}-3D.pt"
        
        if label in self.ns.keys() and not force_rerun:
            return self.ns[label]
        elif os.path.exists(dir_path) and not force_rerun:
            print("[INFO] Loading", dir_path)
            n = torch.load(dir_path)
            self.ns[label] = n
            return n
        
        print("[INFO] Calculating", label)
        print("[INFO] dir_path", dir_path)
        
        ## Init empty direction 
        q = self.sd.sample(etas=self.etas)
        n = Q()
        if self.sd.is_vq:
            n.wT = torch.zeros_like(q.wT)
        else:
            n.xT = torch.zeros_like(q.xT)
        n.x0 = torch.zeros_like(q.x0)
        n.hs = torch.zeros_like(q.hs)
        n.delta_hs = torch.zeros_like(q.hs)
        if not self.etas is None: 
            n.zs = torch.zeros_like(q.zs)

        neg_fs = self.get_paths("neutral")
        pos_fs = self.get_paths(label)
            
        num_samples = len(pos_fs)
        # Calculate Direction   
        for pos_f,neg_f in tqdm(zip(pos_fs, neg_fs)):

            img_pos =  self.load_and_resize(pos_f)
            img_neg =  self.load_and_resize(neg_f)

            q_pos = Q(x0=img_pos, etas=self.etas)
            q_pos = self.sd.encode(q_pos)
            q_neg = Q(x0=img_neg, etas=self.etas)
            q_neg = self.sd.encode(q_neg)
    
            if self.sd.is_vq:
                n.wT += (q_pos.wT - q_neg.wT)/num_samples
            else:
                n.xT += (q_pos.xT - q_neg.xT)/num_samples
            if not self.etas is None:
                n.zs += (q_pos.zs - q_neg.zs)/num_samples
            
            n.x0 += (q_pos.x0 - q_neg.x0)/num_samples
            n.delta_hs += (q_pos.hs - q_neg.hs)/num_samples
        self.ns[label] = n 

 
        torch.save(n,dir_path)
        print("[INFO] Saved to", dir_path)
        return n
    
    def make_imgs(self, q, scales, titles = True,  space = "hspace",font_size = 40):
        q_edits = []
        for dir,s in zip(self.dirs, scales):
            print(dir,s)
            n = self.calc_direction(dir)
            q_edit = self.sd.apply_direction(q.copy(),n,scale = s, space=space)
            q_edits.append(q_edit)
        imgs = [q.x0 for q in q_edits]
        
        # titles_ = ["Original"] + self.dirs
        titles_ = ["Original",'Anger', 'Disgust', 'Fear','Happiness', 'Sadness', 'Surprise']

        if titles == False:
            titles_ = None

        return image_grid([q.x0] + imgs, titles= titles_, font_size = font_size, top = font_size)