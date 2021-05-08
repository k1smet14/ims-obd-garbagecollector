import argparse
import warnings
import random
from importlib import import_module
import gc
import os
import wandb

from tqdm import tqdm

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torchvision import models

from loss import create_criterion

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(data_dir, args):
    
    # setting
    killmemory()
    seed_everything(args.seed)
    warnings.filterwarnings(action='ignore')
    create_dir('saved_models')
    create_dir(f'saved_models/{args.model_name}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # dataset
    dataset_module = getattr(import_module('dataset'), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir,
        age_filter=args.age_filter,
        val_ratio=args.p_val,
        seed=args.seed
    )
    num_classes = dataset.num_classes

    # augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module(
        augp=args.aug_p, 
        resize=args.resize, 
        mean=dataset.mean, 
        std=dataset.std
    )
    dataset.set_transform(transform)

    # dataloader
    train_sampler, val_sampler = dataset.split_dataset()

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=0,
        sampler=val_sampler
    )

    # model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes=num_classes
    ).to(device)
    # model = torch.nn.DataParallel(model) # using mutiple GPU for parallelism
    
    # loss & optimizer & lr scheduler
    criterion = create_criterion(args.criterion)

    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    sched_module = getattr(import_module("torch.optim.lr_scheduler"), args.lr_scheduler)
    scheduler = sched_module(
        optimizer,
        step_size = args.lr_decay_step,
        gamma=0.5
    )

    # training
    print('* Training...')
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss_list = []
        train_acc_list = []

        model.train()
        for x, y in tqdm(train_loader):
            train_x = x.to(device)
            train_y = y.to(device)

            train_pred = model(train_x)
            train_loss = criterion(train_pred, train_y)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            _, train_pred_class = torch.max(train_pred, dim=1)
            train_acc = torch.sum(train_pred_class == train_y).item()/len(train_y)

            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc)
        
        if (epoch % args.print_every==0) or (epoch==args.epochs-1):
            val_loss_list = []
            val_acc_list = []

            with torch.no_grad():
                model.eval()
                for x, y in val_loader:
                    val_x = x.to(device)
                    val_y = y.to(device)

                    val_pred = model(val_x)
                    val_loss = criterion(val_pred, val_y)

                    _, val_pred_class = torch.max(val_pred, 1)
                    val_acc = torch.sum(val_pred_class == val_y).item()/len(val_y)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)
            
            mean_train_loss = np.mean(train_acc_list)
            mean_train_acc = np.mean(train_acc_list)
            mean_val_loss = np.mean(val_loss_list)
            mean_val_acc = np.mean(val_acc_list)
            current_lr = get_lr(optimizer)

            print(
                f'* Epoch : {epoch} (current lr : {current_lr}) \n'
                f'mean train loss : {mean_train_loss:.3f}, mean train acc : {mean_train_acc:.3f} \n'
                f'  mean val loss : {mean_val_loss:.3f},   mean val acc : {mean_val_acc:.3f} \n'
            )

            wandb.log({
                "epoch" : epoch, 
                "mean_train_loss" : mean_train_loss,
                "mean_train_acc" : mean_train_acc,
                "mean_val_loss" : mean_val_loss,
                "mean_val_acc" : mean_val_acc
            })
        
            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                save_path = f'saved_models/{args.model_name}/{epoch:0>2}_{mean_train_acc:.3f}_{mean_val_acc:.3f}.pth'
                torch.save(model.state_dict(), save_path)
                print(f'Saved models : {save_path} \n')


if __name__ == '__main__':
    # initializing wandb
    wandb.init(project='stage1', entity='doooom')
    
    # get argumnet
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument('--age_filter', type=int, default=60)
    parser.add_argument('--p_val', type=float, default=0.3)
    parser.add_argument('--dataset', type=str, default='RandomDataset')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation')
    parser.add_argument('--aug_p', type=float, default=0.5)
    parser.add_argument('--resize', type=tuple, default=(512,384))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=2)
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--criterion', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=1)

    parser.add_argument('--data_dir', type=str, default='D:\\stage1\\input\\data\\train\\images')

    args = parser.parse_args()

    # get directory name same as wandb row name
    model_name = input('Write model name (Syncing run) : ')
    args.model_name = model_name
    print(args)

    # wandb update args
    wandb.config.update(args)
    
    # training
    data_dir = args.data_dir
    train(data_dir, args)