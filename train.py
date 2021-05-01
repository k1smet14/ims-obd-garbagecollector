import os
import random
import json
import warnings 
import time
import gc
from tqdm import tqdm
import argparse
from importlib import import_module
from easydict import EasyDict
import wandb

import torch
from torch.utils.data import DataLoader
from utils import label_accuracy_score

import numpy as np
import pandas as pd


def killmemory():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    # setting 
    killmemory()
    seed_everything(args.seed)
    warnings.filterwarnings(action='ignore')
    create_dir('./saved_model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # augmentation
    train_transform_module = getattr(import_module("augmentation"), args.augmentation)
    train_transform = train_transform_module(args.augp, args.resize)
    val_transform_module = getattr(import_module("augmentation"), 'ValAugmentation')
    val_transform = val_transform_module(args.resize)
    
    # dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    train_dataset = dataset_module(data_dir=args.train_path, mode='train', transform=train_transform)
    val_dataset = dataset_module(data_dir=args.val_path, mode='val', transform=val_transform)

    # data loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=12)
    model.to(device)

    # training
    print('* Start Training...')

    #criterion_module = getattr(import_module("torch.nn"), args.loss)
    # custom loss를 사용하고 싶으면 아래처럼 사용
    criterion_module = getattr(import_module("loss"), args.loss)   
    criterion = criterion_module()

    optimizer_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = optimizer_module(params = model.parameters(), lr=args.learning_rate)

    scheduler_module = getattr(import_module("scheduler"), args.scheduler)
    scheduler = scheduler_module(optimizer, T_0=args.epochs, eta_max=args.max_learning_rate, T_up=2, gamma=0.5)

    best_loss = 9999999
    for epoch in range(args.epochs):
        print('-' * 80)
        print(f'* Epoch {epoch+1}')
        start_time = time.time()
        
        model.train()
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)        # (batch, channel, height, width)
            masks = torch.stack(masks).long()   # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)

            outputs = model(images).to(device)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1)%25==0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f} , LR:{scheduler.get_lr()[0]:6f}')

        scheduler.step()

        if (epoch+1) % args.val_every == 0:
            print()
            avg_loss, avg_mIoU = validation(model, val_loader, criterion, device)
            train_time = time.time()-start_time
            print(f"\n* epoch {epoch+1} training and validation time : {train_time:.4f} sec \n")

            # wandb logging
            wandb.log({"time": train_time, "train_loss": loss.item(), "val_loss": avg_loss, "val_mIoU": avg_mIoU})
            
            if avg_loss < best_loss:
                print(f'Best performance at epoch {epoch+1}')
                print(f'Save model in saved_model/{args.save_file_name}.pt \n')
                best_loss = avg_loss
                save_model(model, './saved_model', f'{args.save_file_name}.pt')


def validation(model, data_loader, criterion, device):
    print('* Start validation...')
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images).to(device)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            mIoU_list.append(mIoU)
            
        avrg_loss = total_loss / cnt
        print(f'Validation Average Loss: {avrg_loss:.4f}, mIoU: {np.mean(mIoU_list):.4f}')

    return avrg_loss, np.mean(mIoU_list)


def save_model(model, saved_dir, file_name):
    # check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path) 


if __name__=='__main__':
    # wandb initializing
    wandb.init(project='stage3', entity='doooom')
    
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get args in config file
    args = EasyDict()
    with open(f'./config/{ipts.config_name}.json', 'r') as f:
        args.update(json.load(f))

    # save hyperparameters in wandb
    wandb.config.config_name = ipts.config_name
    wandb.config.update(args)

    # training
    train(args)
    