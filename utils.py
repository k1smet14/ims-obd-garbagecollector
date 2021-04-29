import os

import numpy as np
import torch

from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

import segmentation_models_pytorch as smp


def save_model(model, saved_dir="model", file_name="default.pt"):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    check_point = {'model' : model.state_dict()}
    path = os.path.join(saved_dir, file_name)
    torch.save(check_point, path)
    
def load_model(model, device, saved_dir="model", file_name="default.pt"):
    path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(state_dict=checkpoint['model'])
    print("load success")

def get_model(args,classes=12):
    if args.network.startswith('swin'):
        repo_root='/opt/ml/p3-ims-obd-garbagecollector'
        if args.network[-1] == 's':
            cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py'))
            cfg.model.pretrained=os.path.join(repo_root,'swin_weight/swin_small_patch4_window7_224.pth')

        elif args.network[-1] == 'b':
            cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py'))
            cfg.model.pretrained=os.path.join(repo_root,'swin_weight/swin_base_patch4_window12_384_22k.pth')
        else:
            cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'))
            cfg.model.pretrained=os.path.join(repo_root,'swin_weight/swin_tiny_patch4_window7_224.pth')

        cfg.model.backbone.use_checkpoint=True
        return build_segmentor(cfg.model,train_cfg=None,
         test_cfg=None)
    elif args.network == 'labv3p':
        return smp.DeepLabV3Plus(
        encoder_name=args.backbone_name,
        encoder_weights='imagenet', 
        classes=12
        )
    
