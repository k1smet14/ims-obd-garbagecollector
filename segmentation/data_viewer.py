import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pycocotools.coco import COCO
import pandas as pd


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def get_img_mask(index, category_names,coco,data_dir):
        # dataset이 index되어 list처럼 동작
        image_id = coco.getImgIds(imgIds=index)
        image_infos = coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        ann_ids = coco.getAnnIds(imgIds=image_infos['id'])
        anns = coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)

        # masks : size가 (height x width)인 2D
        # 각각의 pixel 값에는 "category id + 1" 할당
        # Background = 0
        masks = np.zeros((image_infos["height"], image_infos["width"]))
        # Unknown = 1, General trash = 2, ... , Cigarette = 11
        unique_cls = []
        for i in range(len(anns)):
            className = get_classname(anns[i]['category_id'], cats)
            unique_cls.append(className)
            pixel_value = category_names.index(className)
            masks = np.maximum(coco.annToMask(anns[i])*pixel_value, masks)
        masks = masks.astype(np.float32)
        return images, masks, np.unique(unique_cls)

def auto_viewer(args):
    color_palette={
    '0':np.array([0,0,0]), # white
    '1':np.array([255,255,0]), # Yellow
    '2':np.array([0,255,0]), # Green
    '3':np.array([0,0,255]), # Blue
    '4':np.array([0,255,255]), # sky blue
    '5':np.array([255,0,255]), # pink
    '6':np.array([128,0,255]), # purple
    '7':np.array([128,128,128]), # gray
    '8':np.array([255,128,0]), # orange
    '9':np.array([0,255,128]), # light green
    '10':np.array([255,255,255]), # black
    '11':np.array([255,0,0]) # Red
    }
    json_path = os.path.join(args.data_dir,args.json_file)
    coco = COCO(json_path)
    # Read annotations
    with open(json_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    len_imgs = len(dataset['images'])
    
    # Load categories and super categories
    cat_names = []
    for cat_it in categories:
        cat_names.append(cat_it['name'])
    cat_color = ['Yellow', 'Green','Blue', 'sky' 'blue', 'pink', 'purple', 'gray', 'orange','light green', 'White','Red']

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Color': cat_color})

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    back_pd = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    df =back_pd.append( df,ignore_index=True)
    category_names = list(df.Categories)

    idx = args.start_id
    while True:

        image, mask,clss = get_img_mask(idx,category_names,coco,args.data_dir)

        mask= np.expand_dims(mask,-1)
        mask = np.select([mask==i for i in range(12)],[color_palette[str(i)] for i in range(12)])
        image,mask =np.uint8(image), np.uint8(mask)
        image, mask = cv2.cvtColor(image,cv2.COLOR_BGR2RGB), cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        print("Color Table\n",df)
        print(clss)
        print("Next : 'd', Pre : 'a', Exit : 'q'")
        cv2.imshow("ori",image)
        cv2.imshow("mask",mask)
        
        cv2.imshow("im+mask",np.uint8(image*0.5 + mask*0.5))
        key = cv2.waitKey()
        key = chr(key)
        if key.lower()=='d':
            idx+=1
            idx = min(idx,len_imgs-1)
        elif key.lower()=='a':
            idx-=1
            idx = max(idx,0)
        elif key.lower()=='q':
            break

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Segmentation data viewer')    
    parser.add_argument('--start_id', type=int,default=0)
    parser.add_argument('--json_file',type=str,default='train.json')
    parser.add_argument('--data_dir',type=str,default='./input/data')
    args = parser.parse_args()
    
    auto_viewer(args)
    