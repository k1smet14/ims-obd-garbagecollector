import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 전처리를 위한 라이브러리
import cv2
from pycocotools.coco import COCO
# import torchvision
# import torchvision.transforms as transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

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
        self.dataset_path = '../input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.uint8)
        #images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

    
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))



class PseudoTrainset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, transform = None):
        super().__init__()
        self.dataset_path = '..input/data/'  ### 경로 수정 ###
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
        self.pseudo_imgs = np.load(self.dataset_path+'pseudo_imgs_path.npy')
        self.pseudo_masks = sorted(glob.glob(self.dataset_path+'pseudo_masks/*.npy'))
        
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