import sys
sys.path.insert(0,'./swin')

from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import torch


def main():
    cfg = Config.fromfile('swin/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py')
    cfg.model.pretrained='./swin_weight/swin_base_patch4_window12_384_22k.pth'
    cfg.model.backbone.use_checkpoint=True
    
    model = build_segmentor(cfg.model,train_cfg=None,
         test_cfg=None)
    print(model)
    print("Swin Model load Success")


if __name__ == '__main__':
    main()