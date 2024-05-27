import os
import sys
import math
from tqdm import tqdm 
import gdown
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import models

from tqdm import tqdm 

from utils import image_grid
from PIL import Image

from semanticdiffusion import Q


URL_TEMPLATE = 'https://hanlab18.mit.edu/projects/anycost-gan/files/{}_{}.pt'
attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
             'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
             'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
       
class AnycostPredictor:
    """
    Code is adopted from: AnyCostGAN (https://github.com/mit-han-lab/anycost-gan)
    """
    def __init__(self, device = "cuda"):
        self.device = device
        self.estimator = get_pretrained('attribute-predictor').to(self.device) 
        # self.estimator.eval()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def get_attr(self, img):
        # get attribute scores for the generated image
        img = self.face_pool(img)
        logits = self.estimator(img).view(-1, 40, 2)[0]
        attr_preds = torch.nn.functional.softmax(logits,dim = 1).detach().cpu()#.numpy()        
        return attr_preds

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def safe_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False,
                                  file_name=None):

    try:
        import horovod.torch as hvd
        world_size = hvd.size()
    except:  # load horovod failed, just normal environment
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)

    if world_size == 1:
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    else:  # world size > 1
        if hvd.rank() == 0:  # possible download... let it only run on worker 0 to prevent conflict
            _ = torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
        hvd.broadcast(torch.tensor(0), root_rank=0, name='dummy')
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    # return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)

def load_state_dict_from_url(url, key=None):
    if url.startswith('http'):
        sd = safe_load_state_dict_from_url(url, map_location='cpu', progress=True)
    else:
        sd = torch.load(url, map_location='cpu')
    if key is not None:
        return sd[key]
    return sd

def get_pretrained(model, config=None):
    if model in ['attribute-predictor', 'inception']:
        assert config is None
        url = URL_TEMPLATE.format('attribute', 'predictor')  # not used for inception
    else:
        assert config is not None
        url = URL_TEMPLATE.format(model, config)

    if model == 'attribute-predictor':  # attribute predictor is general
        predictor = models.resnet50()
        predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
        predictor.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return predictor
    else:
        raise NotImplementedError

class PoseEstimator(nn.Module):
    """
    HopeNet is adopted from https://github.com/natanielruiz/deep-head-pose/blob/master/code/hopenet.py
    Code is adapted from
    https://github.com/yuval-alaluf/stylegan3-editing/blob/main/editing/interfacegan/helpers/pose_estimator.py
    https://github.com/yuval-alaluf/stylegan3-editing/blob/main/editing/interfacegan/generate_latents_and_attribute_scores.py

    """
    def __init__(self, 
                 model_path = 'pretrained_models/hopenet_robust_alpha1.pkl',
                 device = None):
        if device is None:
            device = "cuda"
    
        super(PoseEstimator, self).__init__()
        self.pose_net = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        self.assure_model(model_path)
        saved_state_dict = torch.load(model_path)
        self.pose_net.load_state_dict(saved_state_dict)
        self.pose_net.to(device)
        self.pose_net.eval()
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(device)

    def assure_model(self, model_path):

        if not os.path.exists(model_path):
            print("Downloading HopeNet pose classifier")
            id = "1xabKxxkghyE2tt6CcgyKDcSIyenOdSBS"
            gdown.download(id = id, output = model_path, quiet=False, fuzzy=True)

    def extract_pose(self, x):
        tensor_img = transforms.Resize(224)(x)
        yaw, pitch, roll = self.pose_net(tensor_img)
        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = softmax_temperature(yaw.data, 1)
        pitch_predicted = softmax_temperature(pitch.data, 1)
        roll_predicted = softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        return yaw_predicted, pitch_predicted, roll_predicted

    def extract_yaw(self, x):
        yaw, pitch, roll = self.extract_pose(x)
        return yaw


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll


class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x

class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll
    

class AnycostDirections:
    
    def __init__(self, sd, 
                 out_folder = "results/anycost/",
                 etas = None,
                 num_examples = 100,
                 idx_size = 10000,
                 balance_gender = False):
        self.out_folder = out_folder
        os.makedirs(self.out_folder,exist_ok=True)
        self.num_examples  = num_examples
        self.sd = sd
        self.device = sd.device
        self.attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.attr_idx = {a:i for i,a in enumerate(self.attr_list)}
        self.pose_labels = ["yaw", "pitch", "roll"]
        self.pose_idx = {a:i for i,a in enumerate(self.pose_labels)}        
        self.idx_size = idx_size
        
        self.test_labels = ["Smiling","Eyeglasses","Male", "Young","Black_Hair",
                            "Bald","Wearing_Hat","Blurry","yaw", "pitch", "Blond_Hair" ]
        self.conf_labels = ["Smiling","Eyeglasses","Male", "Young", "yaw"]
        #,"Young","Blurry"] 
        self.ap = AnycostPredictor(device =self.device)# .to(self.device)
        self.pose_estimator = PoseEstimator(device = self.device)
        self.etas = etas
        self.idx_path = self.out_folder + self.sd.model_label + f"-etas{self.etas}-anycost_idxsize{self.idx_size}.pt"
        self.attrs = self.get_attrs(etas=self.etas)
        self.balance_gender = balance_gender
        self.ns = {}
        
    def compute_test_directions(self):
        for label in self.test_labels:
            self.calc_direction(label)
            
    def compute_conf_directions(self):
        for label in self.conf_labels:
            self.calc_direction(label)
            
    def get_attrs(self,
                force_rerun = False, 
                etas = None):
        
        if os.path.exists(self.idx_path)and not force_rerun:
            print("[INFO] Anycost attr idx loaded from",self.idx_path)
            return torch.load(self.idx_path)       
        
        print("[INFO] Calculating index to", self.idx_path)
        attrs = torch.zeros((self.idx_size,40,2))
        poses = torch.zeros((self.idx_size,3))
        
        for i in tqdm(range(self.idx_size)):
            q = self.sd.sample(seed = i, etas = etas)
            attrs[i] = self.ap.get_attr(q.x0.float())
            poses[i] = torch.cat(self.pose_estimator.extract_pose(q.x0.float()))
        torch.save([attrs, poses], self.idx_path)
        print("[INFO] Anycost attr saved to",self.idx_path)
        return attrs, poses


    def get_attr_values(self, label):

        
        if label in self.pose_labels:
            attr_idx = self.pose_labels.index(label)
            value = self.attrs[1][:,attr_idx]
        else:
            attr_idx = self.attr_list.index(label)
            value = self.attrs[0][:,attr_idx][:,1]
        return value


    def get_idx_for_attr(self, label):

        attr_values = self.get_attr_values(label)

        if self.balance_gender and not label == "Male":
            return self.get_balanced_genders(attr_values)
        
        sort_idx = torch.argsort(attr_values)
        neg_idx = sort_idx[:self.num_examples]
        pos_idx = sort_idx[-self.num_examples:]
        return pos_idx, neg_idx
    
    def set_balance_gender(self, val):
        assert val in [True, False]
        self.balance_gender = val 
        self.ns = {}

    def get_balanced_genders(self, attr_values,
                        thresh_man = 0.19, # above this: assume male
                        thresh_woman = 0.05 # below this: assume women
                       ):
     
        gender = self.get_attr_values("Male")        
        sort_idx = torch.argsort(attr_values)

        neg_idx = []
        imen=0
        iwomen=0
        # for i in range(len(sort_idx)):
            # idx = sort_idx[i]
        for idx in sort_idx:
            bool_man = gender[idx]>thresh_man # try to catch the men
            bool_woman = gender[idx]<thresh_woman # try to catch the women
            if bool_man and imen<int(self.num_examples // 2 ):
                neg_idx.append(int(idx))
                imen+=1
            if bool_woman and iwomen<int(self.num_examples // 2 ):
                neg_idx.append(int(idx))
                iwomen+=1
            if imen+iwomen>self.num_examples:
                break

        pos_idx = []
        imen=0
        iwomen=0
        # loop in reverse direction for positive direction
        # for i in range(len(sort_idx)-1,0,-1):
        #     idx = sort_idx[i]
        for idx in reversed(sort_idx):
            bool_man = gender[idx]>thresh_man # try to catch the men
            bool_woman = gender[idx]<thresh_woman # try to catch the women
            if bool_man and imen<int(self.num_examples // 2 ):
                pos_idx.append(int(idx))
                imen+=1
            if bool_woman and iwomen<int(self.num_examples // 2 ):
                pos_idx.append(int(idx))
                iwomen+=1
            if imen+iwomen>self.num_examples:
                break

        return torch.tensor(pos_idx), torch.tensor(neg_idx)

    def calc_direction_(self, label):
        print("[INFO] Calculating", label)

        pos_idx, neg_idx = self.get_idx_for_attr(label)
        num_samples = len(pos_idx)
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
            
        # n.x0 = torch.zeros_like(q.x0)    
        for seed_pos,seed_neg in tqdm(zip(pos_idx, neg_idx)):
            q_pos = self.sd.sample(seed=seed_pos, etas = self.etas)
            q_neg = self.sd.sample(seed=seed_neg, etas = self.etas)
            if self.sd.is_vq:
                n.wT += (q_pos.wT - q_neg.wT)/num_samples
            else:
                n.xT += (q_pos.xT - q_neg.xT)/num_samples
            if not self.etas is None:
                n.zs += (q_pos.zs - q_neg.zs)/num_samples
            
            n.x0 += (q_pos.x0 - q_neg.x0)/num_samples
            n.delta_hs += (q_pos.hs - q_neg.hs)/num_samples
        return n

    def calc_direction(self, label, force_rerun = False):
        dir_path = self.out_folder + self.sd.model_label 
        dir_path += f"-etas{self.etas}-anycost_idxsize{self.idx_size}-label{label}-numexamples{self.num_examples}-balance_gender{self.balance_gender}.pt"
        
        if label in self.ns.keys() and not force_rerun:
            return self.ns[label]
        elif os.path.exists(dir_path) and not force_rerun:
            print("[INFO] Loading", dir_path)
            n = torch.load(dir_path)
            self.ns[label] = n
            return n

        n = self.calc_direction_(label)    

        self.ns[label] = n 
        # os.remove(dir_path)
        torch.save(n,dir_path)
        print("[INFO] Saved to", dir_path)
        return n

    def plot_test_directions(self, q):
        imgs =[self.sd.interpolate_direction(q, self.calc_direction(label), 
                    space = "hspace", t1 = -1, t2 = 1, 
                    numsteps = 5) for label in self.test_labels]
        return image_grid(imgs, titles = self.test_labels, rows = len(imgs), cols = 1)


    def get_direction(self, label, clabels = None):
        
        """
        Get conditional direction
        """
        
        n = self.calc_direction(label).delta_hs.clone()
        n_perp = n.clone()
        if not clabels is None:
            for clabel in clabels:
                nc = self.calc_direction(clabel).delta_hs.clone()
                n = n - self.sd.inner_product(n,nc) / self.sd.inner_product(nc,nc) * nc

            for clabel in clabels:
                nc = self.calc_direction(clabel).delta_hs.clone()    
        return Q(delta_hs=n)

    def get_cond_dir(self, label, clabels):
        """
        following 
        https://github.com/genforce/interfacegan/blob/8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb/utils/manipulator.py#L190
        """
        primal = self.get_direction(label).delta_hs
        if clabels is None or clabels == []:
            return Q(delta_hs = primal)
        
        primal_shape = primal.shape 
        primal = primal.flatten()

        N = torch.cat([self.get_direction(l).delta_hs.flatten().unsqueeze(0) for l in clabels])
        A = N @ N.T
        B =  N @ primal

        x = torch.linalg.solve(A, B)

        new = primal - x @ N
        # new = primal - N.T @  torch.linalg.inv(A) @ B (fails on machines with low memory)
        new = new.reshape(primal_shape)
        
        return Q(delta_hs = new)

