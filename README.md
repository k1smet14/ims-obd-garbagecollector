# 송민기

### Swin transformer setup
```bash
cd swin
pip install -e .
apt-get install g++
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html  #it need long time

mkdir swin_weight
cd swin_weight
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth
```

### Swin transformer train
```bash
python p3-ims-obd-garbagecollector/train.py --project_name [your project name] --network [swin_t,swin_s,swin_b]
```

### TroubleShoot
- libGL.so.1: cannot open shared object file: No such file or directory
```bash
apt-get install libgl1-mesa-glx
```

