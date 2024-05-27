import os
import cv2 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from functools import partial
from utils import to_np_image




resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class SegmentationNetwork():
    def __init__(self, 
                 n_classes = 19,
                 state_dict_path = "pretrained_models/79999_iter.pth", 
                 device = "cuda"):

        self.n_classes = n_classes
        self.device = device
        self.state_dict_path = state_dict_path
        self.download_model()
        self.net = BiSeNet(n_classes = self.n_classes).eval().to(self.device)
        self.net.load_state_dict(torch.load(self.state_dict_path))
        self.atts = ["background",'skin', 'l_brow', 
                    'r_brow', 'l_eye', 'r_eye',
                     'eye_g', 'l_ear', 'r_ear', 
                     'ear_r','nose','mouth', 
                     'u_lip', 'l_lip', 'neck', 
                     'neck_l', 'cloth', 'hair', 'hat'] 
        
        self.d = {i:j for i,j in zip(self.atts, range(len(self.atts)))}

        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def download_model(self):

        import gdown
        # BISENET_PATH = "pretrained_models/79999_iter.pth"
        # Download Bisenet CelebA mask https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view
        if os.path.exists(self.state_dict_path):
            print("[INFO] model exists:", self.state_dict_path)
        else:
            id = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            gdown.download(id = id, output = self.state_dict_path,
                        quiet=False, fuzzy=True)
    def forward(self,img):
        img = self.transform(img)
        out = self.net(img)[0]
        return out 

    def probability_mass(self,img):
        out = self.forward(img)
        out = torch.nn.functional.softmax(out, dim = 1)

        ## Correct eyes
        # outl_reg = (out[:,self.d["l_eye"]] - out[:,self.d["r_eye"]]).clamp(min = 0, max = 1)
        # outr_reg = (out[:,self.d["r_eye"]] - out[:,self.d["l_eye"]]).clamp(min = 0, max = 1)

        # out[:,self.d["l_eye"]] = outl_reg
        # out[:,self.d["r_eye"]] = outr_reg
        # a = torch.zeros_like(out)
        # a[:,self.d["l_eye"]] = outl_reg - out[:,self.d["l_eye"]]
        # a[:,self.d["r_eye"]] = outr_reg - out[:,self.d["r_eye"]]
        # out += a.clone()
        return out

    def prediction(self, img):
        return self.probability_mass(img).argmax(1)
        
    def mass(self, img):
        out = self.probability_mass(img)
        # out = torch.nn.functional.softmax(out,dim = 1)
        out = out.sum(axis = (2,3))/(512*512)
        return out #self.forward(img)

    def center_off_mass(self, img, idx = 1 ):
        prob_mass = self.probability_mass(img)
        prob_mass = prob_mass[0,idx]
        prob_mass = prob_mass / prob_mass.sum()
        x = prob_mass.sum(axis = 0) @ torch.arange(0,512, dtype=(torch.float32)).to(self.device)
        y = prob_mass.sum(axis = 1) @ torch.arange(0,512, dtype=(torch.float32)).to(self.device)        
        cm = torch.cat([x.unsqueeze(0),y.unsqueeze(0)])
        
        # std_x = prob_mass.sum(axis = 0).std()
        # std_y = prob_mass.sum(axis = 1).std()

        std_x = torch.sqrt(prob_mass.sum(axis = 0) @ (torch.arange(0,512, dtype=(torch.float32)).to(self.device) - x)**2)
        std_y = torch.sqrt(prob_mass.sum(axis = 1) @ (torch.arange(0,512, dtype=(torch.float32)).to(self.device) - x)**2)

        std = torch.cat([std_x.unsqueeze(0),std_y.unsqueeze(0)])
        return cm, std

    def get_atts(self, img):
        """
        returns total mass, normalized enter of mass and standard deviation
        shape (batch, class [mass, cm_x, cm_y, std_x, std_y])
        """
        outs = self.probability_mass(img)

        mass = outs.sum(axis = (2,3))/(512*512)
                        # cm, std = self.center_off_mass(img, idx = idx)
                        # cm, std = self.center_off_mass(img, idx = d["l_lip"])

        ## this terrible line normiles the pdfs for each class independently
        norms = outs.sum(axis = (2,3))
        outs_cnormed = torch.cat([torch.cat([(m/n).unsqueeze(0) for m,n in zip(out, norm)]).unsqueeze(0)
                                for out, norm in zip(outs,norms)])

        grid_y, grid_x = torch.meshgrid(torch.linspace(0,511,512), torch.linspace(0,511,512), indexing='ij')
        grid_x, grid_y = grid_x.to(self.device)/512, grid_y.to(self.device)/512
        
        ## Center of mass
        x_cm_x = (outs_cnormed*grid_x.repeat(1,19,1,1)).sum(axis = (2,3))
        x_cm_y = (outs_cnormed*grid_y.repeat(1,19,1,1)).sum(axis = (2,3))
        x_cms = torch.cat([x_cm_x.unsqueeze(-1),x_cm_y.unsqueeze(-1)], axis = 2)

        ## Standard deviation
        std_x = torch.cat([torch.cat([(outs_cnorm*(grid_x - x_cm[0])**2).sum().unsqueeze(0) for outs_cnorm,x_cm, in zip(outs_cnorms,xs_cm)]).unsqueeze(0) for outs_cnorms, xs_cm in zip(outs_cnormed,x_cms)])
        std_y = torch.cat([torch.cat([(outs_cnorm*(grid_y - x_cm[1])**2).sum().unsqueeze(0) for outs_cnorm,x_cm, in zip(outs_cnorms,xs_cm)]).unsqueeze(0) for outs_cnorms, xs_cm in zip(outs_cnormed,x_cms)])
        std = torch.sqrt(torch.cat([std_x.unsqueeze(-1), std_y.unsqueeze(-1)],axis = 2))

        attrs = torch.cat([mass.unsqueeze(-1), x_cms, std],axis = 2)

        return attrs 
    def get_mask_(self,img,label, size = 256):

        if isinstance(label,str):
            label = self.d[label]
        mask = (self.prediction(img) == label).int().repeat(1,3,1,1)
        mask = torchvision.transforms.Resize((size,size))(mask)
        return mask

    def get_mask(self, q, label):
        if label == "ear":
            mask_earl = self.get_mask_(q.x0, "l_ear")
            mask_earr = self.get_mask_(q.x0, "r_ear")
            mask_ear = (mask_earl+mask_earr).clamp(0,1) 
            return mask_ear

        elif label == "eye":
            mask_eyel = self.get_mask_(q.x0, "l_eye")
            mask_eyer = self.get_mask_(q.x0, "r_eye")
            mask_eye = (mask_eyel + mask_eyer).clamp(0,1)
            return mask_eye

        elif label == "hair":
            mask_hair = self.get_mask_(q.x0, "hair")
            return mask_hair
        
        elif label == "mouth":
            mask_mouth_in = self.get_mask_(q.x0, "mouth")
            mask_mouth_u_lip = self.get_mask_(q.x0, "u_lip")
            mask_mouth_l_lip = self.get_mask_(q.x0, "l_lip")
            mask_mouth = (mask_mouth_in + mask_mouth_u_lip + mask_mouth_l_lip).clamp(0,1)
            return mask_mouth

        elif label == "neck":
            mask_neck_ = self.get_mask_(q.x0, "neck")
            mask_neckl = self.get_mask_(q.x0, "neck_l")
            mask_neck = (mask_neck_+mask_neckl).clamp(0,1)
            return mask_neck
        elif label == "brow": 
            mask_browl = self.get_mask_(q.x0, "l_brow")
            mask_browr = self.get_mask_(q.x0, "r_brow")
            mask_brow = (mask_browl + mask_browr).clamp(0,1)
            return mask_brow
        else: 
            mask = self.get_mask_(q.x0, label)
            return mask

class SegmentationNetworkVis(SegmentationNetwork):
    def __init__(self, *args, **kwargs):
        super(SegmentationNetworkVis, self).__init__(*args, **kwargs)

    def plot_density(self, img, idx = 1):
        out = self.probability_mass(img)
        # out = torch.nn.functional.softmax(out,dim = 1)
        plt.imshow(out[0,idx].detach().cpu().numpy())
        plt.axis("off")
        
    def vis_parsing_maps(self, img, idx = None):
        vis_im, im_orig, vis_parsing_anno = self. parsing_maps(img, show_idx = idx)
        plt.imshow(vis_im)
        plt.axis("off")

    def segmentation_map(self, img):
        vis_im, im_orig, vis_parsing_anno = self.parsing_maps(img, show_idx = None)
        vis_parsing_anno
        return vis_parsing_anno

    def parsing_maps(self, im_orig, 
                    stride=1, 
                    show_idx = None):
        """
        From https://github.com/VisionSystemsInc/face-parsing.PyTorch/blob/master/test.py
        """
        # Colors for all 20 parts
        im = self.transform(im_orig)
        parsing_anno = self.net(im)[0].squeeze(0).detach().cpu().numpy().argmax(0)
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        
        im = to_np_image(im)
        im_orig = transforms.Resize((512,512))(im_orig)
        vis_im = to_np_image(im_orig).copy().astype(np.uint8)
        
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            if show_idx is None:
                index = np.where(vis_parsing_anno == pi)
                vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
            else:
                if show_idx == pi:
                    index = np.where(vis_parsing_anno == pi)
                    vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0)
    
        return vis_im, im_orig, vis_parsing_anno, vis_parsing_anno_color

