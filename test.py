import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from my_utils import *
from dataloader import *
#from loss import *

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    return tuple(zip(*batch))


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in tqdm(enumerate(data_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            # resize (256 x 256)
            # temp_mask = []
            # for img, mask in zip(np.stack(imgs), oms):
            #     transformed = transform(image=img, mask=mask)
            #     mask = transformed['mask']
            #     temp_mask.append(mask)

            # oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
            print(f"step:{step+1:3d}/{len(data_loader)}")
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def main():
    dataset_path = '../input/data'
    test_path = dataset_path + '/test.json'

    test_transform = A.Compose([
                            A.Resize(256, 256),
                            ToTensorV2()
                            ])

    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=16,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    model = smp.DeepLabV3Plus(
        encoder_name='resnext50_32x4d',
        encoder_weights='ssl', 
        classes=12
    )
    load_model(model, device, saved_dir="models", file_name="deeplabv3plus_resnext50_32x4d.pt")
    model.to(device)
    #load_model(model, device, saved_dir, file_name)
    # sample_submisson.csv 열기

    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("./submission/deeplabv3plus_resnext50_32x4d.csv", index=False)

if __name__ == '__main__':
    main()