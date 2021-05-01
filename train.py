import os
import random
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torch.distributed as dist
from pycocotools.coco import COCO
import cv2
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from mmcv.parallel import MMDataParallel
from utils import *
from dataloader import *
from scheduler import *
from evaluate import *

def train(config=None):
    
    
    device = 'cuda'
    
    global args
        ### Hyper parameters ###
    SEED = config['SEED']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    LR = config['LR']
    Eta = config['Eta_Max']
    img_size = 256
    OPTIMIZER = config['Optimizer']
    save_model_name = f'{args.project_name}_{img_size}_seed{SEED}_batch{BATCH_SIZE}_LR{LR}_Eta{Eta}'
    accumulation_step = 1
    best_val_mIoU = 0.00
    
    
        ### SEED setting ###
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

        ### Dataset ###
    dataset_path = 'input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.RandomScale ((0.5, 2.0)),
            A.RandomCrop(img_size,img_size),
            A.HorizontalFlip (0.5),
            A.Normalize (mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
        ])

    val_transform = A.Compose([
                          A.Resize(256, 256),
    A.Normalize (mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
                          ])

    
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
     ### Model ###    
    model = get_model(args)
    model = model.to(device)

        ### Train ###
    criterion = nn.CrossEntropyLoss()
    
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    elif OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=LR)
    elif OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=EPOCHS, eta_max=Eta,  T_up=100, gamma=0.5)
    scaler = GradScaler()
    img_metas =[[{
        'img_shape': (img_size, img_size, 3),
        'ori_shape': (img_size, img_size, 3),
        'pad_shape': (img_size, img_size, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
                'flip_direction': 'horizontal'
            }]]
    print("Start training..")
    
    for epoch in range(EPOCHS):
        epoch+=1
        avg_loss = 0
        batch_count = len(train_loader)

        for step, (images, masks) in enumerate(train_loader):
            start = time.time()
            images, masks = images.to(device), masks.long().to(device)
            
            if args.network.startswith('swin'):
                imgs = [images]
                output = model(imgs,img_metas,return_loss=False)
                loss = criterion(output, masks)
                loss.backward()
                if (step+1)%accumulation_step==0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with autocast():
                    output = model(images)
                    loss = criterion(output, masks)
                scaler.scale(loss).backward()
                if (step+1)%accumulation_step==0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            avg_loss += loss.item() / batch_count
            lr = scheduler.get_lr()[0]
            print(f"\rEpoch:{epoch:3d}  step:{step:3d}/{batch_count-1}  time:{time.time() - start:.3f}  LR:{lr:.6f}", end='')
            
        scheduler.step()
        if args.network.startswith('swin'):
            val_loss, val_mIoU = validation_swin(model, val_loader, criterion, device,img_metas)
        else:
            val_loss, val_mIoU = validation(model, val_loader, criterion, device)
        print("val",avg_loss,val_loss,val_mIoU)
        print(f"   loss: {avg_loss:.3f}  val_loss: {val_loss:.3f}  val_mIoU:{val_mIoU:.3f}")
        if best_val_mIoU < val_mIoU:
            save_model(model, saved_dir="./weight", file_name=save_model_name + f'_epoch{epoch}_score{val_mIoU:.3f}.pt')
            best_val_mIoU = val_mIoU
    print("Finish training")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train')    
    parser.add_argument('--project_name', type=str,default='labv3-resnet101')
    # optimizer
    parser.add_argument('--backbone_name',
                        default='resnet101',
                        const='resnet101',
                        nargs='?',
                        choices=['resnet101', 'resnext50_32x4d'],
                        help='resnet101, resnext50_32x4d')
    parser.add_argument('--network', type=str,
                        default='labv3p',
                        const='labv3p',
                        nargs='?',
                        choices=['labv3p', 'swin_s','swin_b','swin_t'],
                        help='labv3p, swin_s, swin_b (base), swin_t (tiny)')
                        
    parser.add_argument('--count',type=int,default=20)
    args = parser.parse_args()

    config = {
        'SEED': 9,
        'BATCH_SIZE' : 8,
        'LR' : 1e-5,
        'Eta_Max':1e-6,
        'EPOCHS' : 20,
        'Optimizer': 'adamw'
    }
    train(config)
