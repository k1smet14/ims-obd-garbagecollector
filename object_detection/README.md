# How to use swin_transformer
1. Step One - install swin_transformer
```bash
cd swin_object_detection
python setup.py develop
```
2. Step Two - download pretrained model
```bash
cd swin_object_detection
mkdir pretrained
cd pretrained
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth

# please read original repo 
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
```
3. Step Three - train with swin_object_detection/train.ipynb