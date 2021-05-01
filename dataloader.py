import os
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
    
    
def get_transform(config):
    transform_list = []
    if(config['size']==256):
        transform_list.append(A.Resize(256, 256))
    if(config['tta']==True):
        transform_list.append(A.HorizontalFlip(p=1.0))
    transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)

class EnsembleDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, transform_config = None):
        super().__init__()
        self.coco = COCO(data_dir)
        self.dataset_path = 'input/data/'
        if transform_config is not None:
            self.transform = get_transform(transform_config)
        
    def __getitem__(self, index: int):
        
        ### Load Imgs ###
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(self.dataset_path+image_infos['file_name'])
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]
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
