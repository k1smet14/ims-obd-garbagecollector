import os
import random
import time
import json
import wandb
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from pycocotools.coco import COCO
import cv2
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from natsort import natsorted
from torch.cuda.amp import GradScaler, autocast

from utils import *
from dataloader import *
#from loss import *
from scheduler import *
from evaluate import *


def train(config=None):
    wandb.init(config=config)
    config = wandb.config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print (f"This notebook use {device}")
    
        ### Hyper parameters ###
    SEED = config.SEED
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    LR = config.LR
    save_model_name = f'{config.project_name}_seed{SEED}_batch{BATCH_SIZE}'
    accumulation_step = 1
    best_val_mIoU = 0.40
    


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
        A.Resize(256, 256),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=test_transform)
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


        ### Model ###    
    model = smp.DeepLabV3Plus(
        encoder_name='resnext50_32x4d',
        encoder_weights='imagenet', 
        classes=12
    ).to(device)
    wandb.watch(model)
    
    
        ### Train ###
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR[1])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=EPOCHS, eta_max=LR[0],  T_up=2, gamma=0.5)
    
    scaler = GradScaler()
    print("Start training..")
    for epoch in range(EPOCHS):
        epoch+=1
        avg_loss = 0
        batch_count = len(train_loader)

        for step, (images, masks) in enumerate(train_loader):
            start = time.time()
            images, masks = images.to(device), masks.long().to(device)

            with autocast():
                output = model(images)
                loss = criterion(output, masks)
            scaler.scale(loss).backward()

            if (step+1)%accumulation_step==0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
            avg_loss += loss.item() / batch_count
            print(f"\rEpoch:{epoch:3d}  step:{step:3d}/{batch_count-1}  time:{time.time() - start:.3f}  LR:{scheduler.get_lr()[0]:.6f}", end='')
            
        scheduler.step()
        val_loss, val_mIoU = validation(model, val_loader, criterion, device)
        print(f"   loss: {avg_loss:.3f}  val_loss: {val_loss:.3f}  val_mIoU:{val_mIoU:.3f}")
        wandb.log({"loss": avg_loss, "val_loss": val_loss, "val_mIoU": val_mIoU})
        if best_val_mIoU < val_mIoU:
            save_model(model, saved_dir="model", file_name=save_model_name + f'_epoch{epoch}_score{val_mIoU:.3f}.pt')
            best_val_mIoU = val_mIoU
    print("Finish training")


def main():
    project_name = 'resnext50 batch_size 탐색'
    count = 20

    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'val_mIoU',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    
    parameters_dict = {
        'SEED': {
            'distribution': 'int_uniform',
            'max': 9999,
            'min': 1,
        },
        'EPOCHS': {
            'value': 18
        },
        'BATCH_SIZE': {
            'values': [4,24,32]
        },
        'LR': {
            'value': (1e-4,2e-6)
        },
        'scheduler': {
            'value': None
        },
        'project_name':{
            'value': project_name
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train, count=count)

if __name__ == '__main__':
    main()
