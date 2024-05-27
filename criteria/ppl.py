import dnnlib
import torch 
import numpy as np 
import tqdm
class PerceptualPathLength:
 
    # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/perceptual_path_length.py
    """Perceptual Path Length (PPL) from the paper "A Style-Based Generator
    Architecture for Generative Adversarial Networks". Matches the original
    implementation by Karras et al. at
    https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py"""


    def __init__(self, device = None, epsilon = 0.1):

        self._feature_detector_cache = dict()
        self.vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        self.epsilon = epsilon
        self.vgg16 = self.get_feature_detector(self.vgg16_url, num_gpus=1, rank=0, verbose=True).to(device)
        self.ts = torch.linspace(0,1-self.epsilon,int(1/self.epsilon))
     

    def get_feature_detector(self, url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
        assert 0 <= rank < num_gpus
        key = (url, device)
        if key not in self._feature_detector_cache:
            is_leader = (rank == 0)
            if not is_leader and num_gpus > 1:
                torch.distributed.barrier() # leader goes first
            with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
                self._feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
            if is_leader and num_gpus > 1:
                torch.distributed.barrier() # others follow
        return self._feature_detector_cache[key]

    def calc_ppl_from_images(self, imgs):
        vgg_feats = self.vgg16(imgs, resize_images=False, return_lpips=True)
        num_images = len(imgs)
        epsilon = 1/num_images
        print(epsilon)
        dists = []
        for i in range(num_images-1):
                lpips_t0, lpips_t1 = vgg_feats[i], vgg_feats[i+1]
                dist = (lpips_t0 - lpips_t1).square() / epsilon ** 2
                dists.append(dist)

        dist = torch.cat(dists).cpu().numpy()
        lo = np.percentile(dist, 1, method='lower')
        hi = np.percentile(dist, 99, method='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return ppl

    def slerp(self, v0, v1,t,DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        # v0, v1 = v0.numpy(), v1.numpy()
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

    def calc_dist(self,x1,x2,t, decoder = None):
        if decoder is None: decoder = lambda x : x 
        xt0  = self.slerp(x1,x2,t)
        xt1 =  self.slerp(x1,x2,t + self.epsilon)
        imgs = torch.cat([decoder(x) for x in [xt0,xt1]])
        lpips_t0, lpips_t1 = self.vgg16(imgs, resize_images=False, return_lpips=True).chunk(2)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        print(self.epsilon)

        return dist


    def calcualte_ppl(self, x1,x2, decoder = None):
        
        dists = torch.cat([self.calc_dist(x1,x2,t, decoder = decoder) for t in tqdm(self.ts)])
        dist = dists.cpu().numpy()
        lo = np.percentile(dist, 1, method='lower')
        hi = np.percentile(dist, 99, method='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return ppl


