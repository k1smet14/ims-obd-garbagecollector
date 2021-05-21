#%%

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

from tqdm import tqdm

from pycocotools.coco import COCO
import cv2
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from adamp import AdamP

import matplotlib.pyplot as plt
# from natsort import natsorted
from torch.cuda.amp import GradScaler, autocast

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from unet import UNet3Plus, UNet3Plus_DeepSup, UNet3Plus_DeepSup_CGM, UNet3Plus_efficientnet_DeepSup_CGM, UNet3Plus_efficientnet, UNet3Plus_resnext50_32x4d
from unet.efficientunet import *
from unet.efficientnet import *
import timm

from my_utils import *
from dataloader import *
# from loss import *
from scheduler import *
from evaluate import *


def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    wandb.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 8   # Mini-batch size
    num_epochs = 20
    learning_rate = 5e-5

    # seed 고정
    random_seed = 77
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


    # train.json / validation.json / test.json 디렉토리 설정
    dataset_path = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'

    #mean, stds of train_all.json 
    mean=(0.460, 0.440, 0.418)
    std=(0.211, 0.208, 0.216)

    train_transform = A.Compose([
                                A.Resize(256, 256),
                                #A.HorizontalFlip(p=0.5),
                                # A.VerticalFlip(p=0.5),
                                #A.RandomRotate90(p=0.5),
                                #A.CLAHE(p=0.5),
                                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            A.Resize(256, 256),
                            # A.CLAHE(p=1.0),
                            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                            ToTensorV2()
                            ])

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    #saved_dir
    val_every = 1 
    saved_dir = './models'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    # model
    # model = smp.Unet(
    #     encoder_name='timm-efficientnet-b5',
    #     encoder_weights='noisy-student', 
    #     classes=12
    # )
    # model = smp.DeepLabV3Plus(
    #     encoder_name='timm_-agenet',
    #     classes=12
    # )
    # model = smp.UnetPlusPlus(
    #     encoder_name='timm-efficientnet-b0',
    #     encoder_weights='noisy-student',
    #     classes=12
    # )

    # encoder = EfficientNet.encoder('efficientnet-b5', pretrained=True)
    # model = UNet3Plus_efficientnet(encoder, n_classes=12)

    # encoder = timm.create_model('swsl_resnext50_32x4d', pretrained=True)
    # model = UNet3Plus_resnext50_32x4d(encoder, n_classes=12)
    model = UNet3Plus_resnext50_32x4d(n_classes=12)

    model.to(device)
    wandb.watch(model)

    calculate_parameter(model)


    # train_loader의 output 결과(image 및 mask) 확인  
    # for imgs, masks, image_infos in train_loader:
    #     image_infos = image_infos[0]
    #     temp_images = imgs
    #     temp_masks = masks
    #     break

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    # print('image shape:', list(temp_images[2].shape))
    # print('mask shape: ', list(temp_masks[2].shape))
    # # print('Unique values, category of transformed mask : \n', [{int(i),category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])

    # ax1.imshow(temp_images[2].permute([1,2,0]))
    # ax1.grid(False)
    # ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize = 15)

    # ax2.imshow(temp_masks[2])
    # ax2.grid(False)
    # ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize = 15)

    # plt.show()
    # return 0

    #tain
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamP(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=500, max_lr=5e-5, min_lr=5e-7, warmup_steps=100)
    #scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 300, 6540, 3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10000)

    # scaler = GradScaler()
    print('Start training..')
    best_loss = 9999999
    best_mIoU = 0.0
    for epoch in range(num_epochs):
        
        model.train()

        for step, (images, masks, _) in tqdm(enumerate(train_loader)):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            # with autocast():
                # inference
            outputs = model(images)
                # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LR : {:.6f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item(), scheduler.get_lr()[0]))
                    #epoch+1, num_epochs, step+1, len(train_loader), loss.item(), learning_rate))
            wandb.log({'LR': scheduler.get_lr()[0]})
            scheduler.step()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_loss , _ , val_mIoU, val_mIoU2 = validation3(epoch + 1, model, val_loader, criterion, device)
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "val_mIoU": val_mIoU, "val_mIoU2": val_mIoU2})
            # if avrg_loss < best_loss:
            #     print('Best performance at epoch: {}'.format(epoch + 1))
            #     print('Save model in', saved_dir)
            #     best_loss = avrg_loss
            #     save_model(model, saved_dir)
            if best_mIoU < val_mIoU2:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = val_mIoU2
                save_model(model, saved_dir, file_name='UNet3Plus_resnext50_32x4d.pt')
        
        print('finish')


def main():
    train()
    # project_name = 'se_resnext50_32x4d'
    # count = 20
    # sweep_config = {
    #     'method': 'bayes'
    # }
    # metric = {
    #     'name': 'val_mIoU',
    #     'goal': 'maximize'   
    # }
    # sweep_config['metric'] = metric
    
    # parameters_dict = {

    #     'BATCH_SIZE': {
    #         'values': [8,16]
    #     },
    #     'LR': {
    #         'value': (1e-5, 5e-6, 1e-6)
    #     },
    #     'project_name':{
    #         'value': project_name
    #     },
    # }
    # sweep_config['parameters'] = parameters_dict

    # sweep_id = wandb.sweep(sweep_config, project=project_name)
    # wandb.agent(sweep_id, train, count=count)


if __name__ == '__main__':
    main()
# %%
