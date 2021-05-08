import os
from glob import glob
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class CustomAugmentation:
    def __init__(self, augp=0.5, resize=(512,384), mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.246), **args):
        self.transform = Compose([
            Resize(resize[0], resize[1], p=1.0),
            # CenterCrop(resize[0], resize[1], p=1.0),
            # HorizontalFlip(p=augp),
            # ShiftScaleRotate(p=augp),
            # RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.1, 0.3), p=augp),
            # GaussNoise(p=augp),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)
    
    def __call__(self, image):
        return self.transform(image=image)


class RandomDataset(Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2
    
    class GenderLabels:
        male = 0
        female = 1
    
    class AgeGroup:
        def __init__(self, age_filter):
            self.age_filter = age_filter
            self.map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) <self.age_filter else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    multi_class_labels = []

    def __init__(self, data_dir, age_filter=58, mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.246), val_ratio=0.3, seed=0):
        self.data_dir = data_dir
        self.age_filter = age_filter
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.seed = seed

        self.transform = None
        self.setup()
    
    def setup(self):
        profiles = glob(f'{self.data_dir}/*')
        for profile in profiles:
            id, gender, race, age = profile.split('\\')[-1].split('_')
            
            for img_path in glob(f'{profile}/*'):
                mask_label = self._file_names[os.path.splitext(img_path)[0].split('\\')[-1]]
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup(self.age_filter).map_label(age)
                
                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.multi_class_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))
    
    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        multi_class_label = self.multi_class_labels[index]

        if self.transform:
            image_transform = self.transform(np.array(image))
        
        return image_transform['image'], multi_class_label

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def calc_statistics(data_dir):
        img_info = {'means':[], 'stds':[]}
        print('* Calculating images mean and std...')

        profiles = glob(f'{data_dir}/*')
        for profile in tqdm(profiles):
            for img_path in glob(f'{profile}/*'):
                image = Image.open(img_path)
                img_info['means'].append(np.array(image).mean(axis=(0,1)))
                img_info['stds'].append(np.array(image).std(axis=(0,1)))
        
        img_mean = np.mean(img_info["means"], axis=0)/255
        img_std = np.mean(img_info["stds"], axis=0)/255
        print(f'RGB mean of image data : {img_mean}')
        print(f'RGB std of image data : {img_std}')

        return img_mean, img_std

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label
    
    @staticmethod
    def denormalize_image(image:torch.tensor, mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.246)):
        img_cp = image.clone()
        img_cp = img_cp.permute(1,2,0).cpu().numpy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def compare_aug_image(self, index):
        original_img = Image.open(self.image_paths[index])
        transform_img = self.denormalize_image(self[index][0])

        fig, ax = plt.subplots(1, 2, figsize=(7,5))
        ax[0].imshow(original_img)
        ax[0].set_title('original image')
        ax[1].imshow(transform_img)
        ax[1].set_title('transform image')
        plt.show()

    def split_dataset(self):
        train_idx, val_idx = train_test_split(
            np.arange(len(self.multi_class_labels)),
            stratify = self.multi_class_labels,
            test_size = self.val_ratio,
            random_state = self.seed
            )

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        return train_sampler, val_sampler
        '''
        how to use?
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        '''


class TestDataset(Dataset):
    def __init__(self, img_paths, resize=(512, 284), mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize[0], resize[1], p=1.0),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))
        return image['image']
    
    def __len__(self):
        return len(self.img_paths)


if __name__=='__main__':
    data_dir = "D:\\stage1\\input\\data\\train\\images"
    # img_mean, img_std = RandomDataset.calc_statistics(data_dir)
    train_dataset = RandomDataset(data_dir)
    train_dataset.set_transform(CustomAugmentation())
    
    # train_test_split(np.arange(100))

    print(train_dataset.split_dataset())
