import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random

# 전처리를 위한 라이브러리
import cv2
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = 'input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
    def __getitem__(self, index: int):
        
        ### Load Imgs ###
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        ### Train Time ###
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            ###  mask 생성  ###
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            ###  augmentation ###
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                
            return images, masks #, image_infos
        
        ### Test Time ###
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
                
            return images #, image_infos
    
    
    def __len__(self):
        return len(self.coco.getImgIds())
    
    

class EnsembleDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, size, transform=None):
        super().__init__()
        self.coco = COCO(data_dir)
        self.dataset_path = 'input/data/'
        self.transform = transform
        if size==256:
            self.Resize = A.Resize(256, 256)
        else:
            self.Resize = None
        self.Flip = A.HorizontalFlip(p=1.0)
        self.ToTensor = ToTensorV2()
        
    def __getitem__(self, index: int):
        
        ### Load Imgs ###
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if self.Resize is not None:
            images = self.Resize(image=images)["image"]
        if self.transform=='flip':
            images = self.Flip(image=images)["image"]
        elif self.transform=='rotate':
            images = cv2.rotate(images, cv2.ROTATE_90_CLOCKWISE)
        elif self.transform=='rotateR':
            images = cv2.rotate(images, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        images = self.ToTensor(image=images)["image"]
        return images
    
    def __len__(self):
        return len(self.coco.getImgIds())


    

    
    
    
labels_W = [0.0,
            0.00781,
            0.0,
            0.0,
            0.00190,
            0.00223,
            0.00205,
            0.0,
            0.0,
            0.0,
            0.02,
            0.00710]

def labelRandomChoice(labels):
    labels = np.unique(labels)
    choice = np.zeros(12).astype(int)
    choice[labels]=[labels]
    choiced_label = random.choices(choice, weights=labels_W, k=1)
    return torch.LongTensor(choiced_label)
    
    
class MixDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = 'input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
    def __getitem__(self, index: int):
        
        ### Load Imgs ###
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        ### Train Time ###
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            ###  mask 생성  ###
            labels = []
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                labels.append(pixel_value)
            masks = masks.astype(np.float32)

            ###  augmentation ###
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                
            return images, masks, labelRandomChoice(labels)
        
        ### Test Time ###
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
                
            return images #, image_infos
    
    def __len__(self):
        return len(self.coco.getImgIds())

    
    
    
class PseudoTrainset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, transform = None):
        super().__init__()
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = 'input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
        self.pseudo_imgs = np.load('input/data/pseudo_imgs_path.npy')
        self.pseudo_masks = sorted(glob.glob(f'input/data/pseudo_masks/*.npy'))
        
    def __getitem__(self, index: int):
        
        ### Train data ###
        if (index < len(self.coco.getImgIds())):
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]

            images = cv2.imread(self.dataset_path+image_infos['file_name'])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            images /= 255.0
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            ###  mask 생성  ###
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)

        ### Pseudo data ###
        else:
            index -= len(self.coco.getImgIds())
            images = cv2.imread(self.dataset_path+self.pseudo_imgs[index])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            images /= 255.0
            masks = np.load(self.pseudo_masks[index])
            
        ###  augmentation ###
        masks = masks.astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            
        return images, masks
    
    def __len__(self):
        return len(self.coco.getImgIds())+len(self.pseudo_imgs)
    
    
    
    
class PseudoKFoldDataset(Dataset):
    """COCO format"""
    def __init__(self, dataset, transform = None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.coco = COCO('input/data/train_all.json')
        self.dataset_path = 'input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
    def __getitem__(self, index: int):
        
        ### load image ###
        image_infos = self.dataset[index]
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        ### Pseudo mask ###
        if image_infos['pseudo']:
            masks = np.load(self.dataset_path+image_infos['mask_path'])
            
        ### Train mask ###
        else:
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)            
        masks = masks.astype(np.float32)

        ###  augmentation ###
        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
        return images, masks
    
    def __len__(self):
        return len(self.dataset)
    
    

    
class KFoldDataset(Dataset):
    """COCO format"""
    def __init__(self, dataset, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.transform = transform
        self.coco = COCO('input/data/train_all.json')
        self.dataset_path = 'input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
                
        
    def __getitem__(self, index: int):
        image_infos = self.dataset[index]
        
        ### load Data ###
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        ### Train Time ###
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            
            ###  mask 생성  ###
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            ###  augmentation ###
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                
            return images, masks
        
        ### Test Time ###
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
                
            return images
    
    
    def __len__(self):
        return len(self.dataset)