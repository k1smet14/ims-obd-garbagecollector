import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
from .swin_transformer import SwinTransformer

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

def get_model(args,img_size,classes=12):
    if args.network.startswith('swin'):
        if args.network[-1] == 's':
            return SwinTransformer(img_size=img_size, # useage
                                patch_size=4, # 4
                                in_chans=3, # 3
                                num_classes=classes, # useage
                                embed_dim=96, # yaml
                                depths=[ 2, 2, 18, 2 ], # yaml
                                num_heads=[ 3, 6, 12, 24 ], # yaml
                                window_size=7, # 7, yaml
                                mlp_ratio=4, # 4
                                qkv_bias=True, # True
                                qk_scale=None, # None
                                drop_rate=0.0, # 0.0
                                drop_path_rate=0.3, # 0.1, yaml
                                ape=False, # False
                                patch_norm=True, # True
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT) # Useage
            return SwinTransformer(img_size=img_size, # useage
                                patch_size=4, # 4
                                in_chans=3, # 3
                                num_classes=classes, # useage
                                embed_dim=128, # yaml
                                depths=[ 2, 2, 18, 2 ], # yaml
                                num_heads=[ 4, 8, 16, 32 ], # yaml
                                window_size=7, # 7, yaml
                                mlp_ratio=4, # 4
                                qkv_bias=True, # True
                                qk_scale=None, # None
                                drop_rate=0.0, # 0.0
                                drop_path_rate=0.5, # 0.1, yaml
                                ape=False, # False
                                patch_norm=True, # True
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT) # Useage
        else
            return
    elif args.network == 'labv3p':
        return smp.DeepLabV3Plus(
        encoder_name=args.backbone_name,
        encoder_weights='imagenet', 
        classes=12
        )
    
