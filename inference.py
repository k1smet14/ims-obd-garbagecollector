import os
import json
from tqdm import tqdm
import argparse
from importlib import import_module
from easydict import EasyDict

import requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import albumentations as A
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def collate_fn(batch):
    return tuple(zip(*batch))

def inference(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get augmentation
    test_transform_module = getattr(import_module("augmentation"), 'TestAugmentation')
    test_transform = test_transform_module(args.resize)

    # get data set
    dataset_module = getattr(import_module("dataset"), args.dataset)
    test_dataset = dataset_module(data_dir=args.test_path, mode='test', transform=test_transform)

    # data loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=12)

    model_path = f'./saved_model/{args.save_file_name}.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # inference
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('* Start inference...')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in tqdm(enumerate(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
   
    file_names = [y for x in file_name_list for y in x]

    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/{args.save_file_name}.csv", index=False)
    print("* End infernece.")


def submit(file_path = '', desc=""):
    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/28/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': 'Bearer fff7af656e768f558293ce6dd0008088ea1ba6b1'
    }

    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000028/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})



if __name__ == '__main__':
    create_dir('./submission')

    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get args in config file
    args = EasyDict()
    with open(f'./config/{ipts.config_name}.json', 'r') as f:
        args.update(json.load(f))
    
    inference(args)

    print('\n* Start submission...')
    submit(f'./submission/{args.save_file_name}.csv', f'{args.save_file_name} / {ipts.config_name}')
    print('* End submission.')