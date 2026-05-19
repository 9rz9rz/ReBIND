import pathlib
import os
from typing import List
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import torch
import logging
import time
import sys
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np
import json
import copy
import random


dir_path = os.path.dirname(os.path.dirname(__file__))
base_path = pathlib.Path(dir_path)

device = torch.device('cuda')


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def set_device(i):
    global device
    
    torch.cuda.set_device(f"cuda:{i}")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{i}')
    else:
        device = torch.device('cpu')
    return
    

def get_closs(i2t, target, loss_name=None):
    loss = torch.tensor(0.).to(i2t.device)
    bs = i2t.shape[0]
    if bs == 0:
        return loss
    if loss_name == 'None' or loss_name is None:
        return loss
    if loss_name == 'RCL':
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), target] = 0.
        loss = - ((1. - i2t).log() * mask).sum() / bs
        return loss
    if loss_name == 'infoNCE':
        mask = torch.zeros_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), target] = 1.
        loss = - (i2t.log() * mask).sum() / bs
        return loss
    raise ValueError('loss name is invalid')

def get_aloss(left, right, loss_name=None):
    bs = left.shape[0]
    loss = torch.tensor(0.).to(left.device)
    mse_criterion = torch.nn.MSELoss()
    sml1_criterion = torch.nn.SmoothL1Loss()
    if bs == 0:
        return loss
    if loss_name is None or loss_name == 'None':
        return loss
    if loss_name == 'MSE':
        loss = mse_criterion(left, right)
        return loss
    if loss_name == 'SmoothL1':
        loss = sml1_criterion(left, right)
        return loss
    raise ValueError('loss name is invalid')

def robust_mse(left, right, labels, pn_loss):
    clean_mask = labels.to(bool)
    noise_mask = ~clean_mask
    ploss = get_aloss(left[clean_mask], right[clean_mask], pn_loss['positive_align_loss'])
    nloss = get_aloss(left[noise_mask], right[noise_mask], pn_loss['negative_align_loss']) 
    trade_off = pn_loss['trade_off_align']
    loss_dca = trade_off * ploss + (1-trade_off) * nloss
    return loss_dca

class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")   

def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def get_log(dataset_name, exp_name):
    if not os.path.exists(base_path /'log'):
        os.mkdir(base_path / 'log')
    if not os.path.exists(base_path / 'log' / dataset_name):
        os.mkdir(base_path / 'log' / dataset_name)
    timestamp = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
    if exp_name:
        log_folder_path = base_path / 'log' / dataset_name / exp_name
    else:
        log_folder_path = base_path / 'log' / dataset_name / timestamp
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    log_format = '%(asctime)s: %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    log_file_path = log_folder_path / 'process.log'
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return log_folder_path, timestamp

def get_log_simple(file_path):
    log_format = '%(asctime)s: %(message)s'
    date_format = '%Y-%m-%d-%H-%M-%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt=date_format)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fh = logging.FileHandler(file_path)
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(fh)
    return file_path

class Params:
    bool_initialize = False
    @staticmethod
    def initialize(args):
        global params
        if Params.bool_initialize:
            print("️️params have been initialized!")
        if args is None:
            raise ValueError('params list should not be None')
        print('Initialize params')
        params.set_params(args)
        Params.bool_initialize = True
    
    def __init__(self):
        pass
    
    def set_params(self, args):
        self.dataset = args.dataset
        self.method = args.method
        self.noise_ratio = args.noise_ratio
        self.nc_type = args.nc_type

        self.save_training = args.save_training
        self.backbone = args.backbone
        self.num_workers = args.num_workers
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.seed = args.seed
        self.shuffle_seed = args.shuffle_seed
        
        self.timestamp = args.timestamp
        self.pn_loss = {}
        self.pn_loss['positive_loss'] = args.positive_loss
        self.pn_loss['negative_loss'] = args.negative_loss
        self.pn_loss['positive_align_loss'] = args.positive_align_loss
        self.pn_loss['negative_align_loss'] = args.positive_align_loss
        self.pn_loss['trade_off'] = args.trade_off
        self.pn_loss['trade_off_align'] = args.trade_off_align
        self.pn_loss['warmup_loss'] = args.warmup_loss
        self.pn_loss['warmup_align_loss'] = args.warmup_align_loss
        
        self.lrm = args.lrm
        self.lpm = args.lpm
        self.lsa = args.lsa
        self.lrd = args.lrd
        
        
        self.warmup_qformer = args.warmup_qformer
        self.warmup_proj = args.warmup_proj
        self.warmup_last = args.warmup_last
        
        self.warmup_epoch = (
            self.warmup_qformer + self.warmup_proj + self.warmup_last
            if self.method == 'image_diff' else
            args.warmup_epoch
        )
        
        self.partitioner = args.partitioner
        self.split_type = args.split_type
        self.threshold = args.threshold
        
        self.exp_name = args.exp_name

        self.inn_rho = args.inn_rho
        self.inn_sinkhorn_reg = args.inn_sinkhorn_reg
        self.inn_sinkhorn_iter = args.inn_sinkhorn_iter
        self.inn_ortho_coef = args.inn_ortho_coef
        self.inn_negative_slope = args.inn_negative_slope
        self.inn_block_dim = args.inn_block_dim

        self.swap_warmup_epochs = args.swap_warmup_epochs
        self.swap_coef_main = args.swap_coef_main
        self.swap_coef_aux1 = args.swap_coef_aux1
        self.swap_coef_aux2 = args.swap_coef_aux2
        self.swap_coef_bce = args.swap_coef_bce

        self.inference_balance = args.inference_balance

        self.similarity_option = args.similarity_option
        self.aux_loss_option = args.aux_loss_option
        self.variant = args.variant

    def __call__(self):
        display_dict = copy.deepcopy(self.__dict__)
        keys_to_remove = ['num_workers', 'timestamp']
        for key in keys_to_remove:
            del display_dict[key]
        return display_dict
    
params = Params()

class Partitioner:
    
    def __init__(self, type, split, threshold=0.5, timestamp=None, epoch=None, dataset_name=None):
        self.type = type
        self.split = split
        self.threshold = threshold
        self.timestamp = timestamp
        self.epoch = epoch
        self.dataset_name = dataset_name
        
    def fit_features(self, model, trainloader, txt_processors):
        dataset = trainloader.dataset
        if self.type == 'all_positive':
            logging.info('no partition, all positive')
            return torch.ones(len(dataset)) # all clean
        logging.info('fitting partitioner...')
        model.eval()
        data_size = len(dataset)
        loss = torch.zeros(data_size)
        sim = torch.zeros(data_size)
        for reference_name, target_name, captions, index in tqdm(trainloader, ncols=150, mininterval=30):
            reference_images = dataset.get_image_features(reference_name).to(device, non_blocking=True)
            target_images = dataset.get_image_features(target_name).to(device, non_blocking=True)
            if self.dataset_name == 'FashionIQ':
                    flattened_captions = np.array(captions).T.flatten().tolist()
                    captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors['eval'](caption) for caption in captions]
            l, s = model.per_loss(reference_images, target_images, captions)
            for b in range(l.size(0)):
                loss[index[b]] = l[b]
                sim[index[b]] = s[b]
        loss_range = loss.max() - loss.min()
        sim_range = sim.max() - sim.min()
        self.losses = (loss - loss.min()) / loss_range if loss_range > 0 else torch.zeros_like(loss)
        self.sims = (sim - sim.min()) / sim_range if sim_range > 0 else torch.zeros_like(sim)
        self.pred = self.get_pred(self.type)
        return self.pred

    # Previous version: not use precomputed features.
    def fit(self, model, trainloader, txt_processors):
        if self.type == 'all_positive':
            logging.info('no partition, all positive')
            return torch.ones(len(trainloader.dataset))
        logging.info('fitting partitioner...')
        model.eval()
        data_size = len(trainloader.dataset)
        loss = torch.zeros(data_size)
        sim = torch.zeros(data_size)
        with tqdm(total=len(trainloader), mininterval=30) as t:
            for i, data in enumerate(trainloader):
                reference_image = data['source_img_data'].to(device, non_blocking=True)
                target_image = data['target_img_data'].to(device, non_blocking=True)
                captions = data['mod']['str']
                if self.dataset_name == 'FashionIQ':
                    flattened_captions = np.array(captions).T.flatten().tolist()
                    captions = generate_randomized_fiq_caption(flattened_captions)
                captions = [txt_processors['eval'](caption) for caption in captions]
                index = data['index']
                l, s = model.per_loss(reference_image, target_image, captions)
                for b in range(l.size(0)):
                    loss[index[b]] = l[b]
                    sim[index[b]] = s[b]
                t.update()
        # self.gt_labels = gt_labels
        loss_range = loss.max() - loss.min()
        sim_range = sim.max() - sim.min()
        self.losses = (loss - loss.min()) / loss_range if loss_range > 0 else torch.zeros_like(loss)
        self.sims = (sim - sim.min()) / sim_range if sim_range > 0 else torch.zeros_like(sim)
        self.pred = self.get_pred(self.type)       
        return self.pred
    
    def get_pred(self, type, threshold=None):
        type = type.lower()
        if threshold is None:
            threshold = self.threshold
        if type.lower() == 'gmm':
            input_loss = self.losses.reshape(-1,1) 
            input_sim = self.sims.reshape(-1,1)
            input_data = input_loss if self.split == 'loss' else input_sim
        
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_data.cpu().numpy())
            clean_component_idx = gmm.means_.argmin() if self.split == 'loss' else gmm.means_.argmax()
            self.prob = torch.Tensor(gmm.predict_proba(input_data.cpu().numpy())[:, clean_component_idx])
            
            self.pred = (self.prob > threshold) + 0
            
            area_num = torch.histc(torch.tensor(self.prob), bins=10, min=0.0, max=1.0).to(torch.int).tolist()
            logging.info(f'The counts in the equal areas are: {area_num}')
            clean_pro = self.pred.sum().item() / self.pred.shape[0]
            logging.info(f'the proportion of clean samples are {clean_pro}')
            return self.pred
        elif type == 'direct':
            if self.split == 'loss':
                input_data = self.losses
            elif self.split == 'sim':
                input_data = self.sims
            else:
                raise ValueError(f"the parameter split is invalid.")
            self.pred = (input_data < threshold) + 0
            self.prob = self.pred
            print('the proportion of clean samples are ', self.pred.sum().item() / self.pred.shape[0])
            return self.pred
        elif type == 'percent':
            if self.split == 'loss':
                input_data = self.losses
            elif self.split == 'sim':
                input_data = self.sims
            else:
                raise ValueError(f"the parameter split is invalid.")
            noisy_indices = input_data.argsort(descending=True)[:int(threshold * input_data.shape[0])]
            self.pred = torch.ones_like(input_data)
            self.pred[noisy_indices] = 0
            self.prob = self.pred
            print('the proportion of clean samples are ', self.pred.sum().item() / self.pred.shape[0])
            return self.pred
        else:
            raise ValueError(f"the parameter type is invalid.")
        
    def get_prob(self):
        if self.prob is None:
            raise KeyError('prob does not exist')
        else:
            return self.prob
        
def custom_json_dumps(data, indent=4):
    def serialize(obj, indent_level=0):
        if isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                items.append(f'\n{" " * indent * (indent_level + 1)}"{key}": {serialize(value, indent_level + 1)}')
            return f'{{{",".join(items)}\n{" " * indent * indent_level}}}'
        elif isinstance(obj, list):
            items = [json.dumps(item, indent=0) for item in obj]
            return f'[{", ".join(items)}]'
        else:
            return json.dumps(obj)

    return serialize(data)