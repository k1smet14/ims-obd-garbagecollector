import argparse
import os
from importlib import import_module
from random import Random
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import RandomDataset, TestDataset

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_model(num_classes, args, device):
    model_module = getattr(import_module('model'), args.model)
    model = model_module(
        num_classes=num_classes
    )

    model_path = f'./saved_models/{args.model_name}/{args.model_state}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, args):
    create_dir(f'submissions/{args.model_name}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    num_classes = RandomDataset.num_classes
    model = load_model(num_classes, args, device)
    model.eval()

    img_root = data_dir
    info_path = 'D:/stage1/info.csv'
    submission = pd.read_csv(info_path)

    img_paths = [f'{img_root}/{img_id}' for img_id in submission.ImageID]
    dataset = TestDataset(img_paths, resize=args.resize)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )

    print("* Calculating inference results...")
    preds = []
    with torch.no_grad():
        for img in tqdm(dataloader):
            img = img.to(device)
            pred = model(img)
            _, pred_class = torch.max(pred, 1)
            preds.extend(pred_class.cpu().numpy())
    
    submission['ans'] = preds
    submission.to_csv(f"submissions/{args.model_name}/{args.model_state}.csv", index=False)
    print('* Inference Done!')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_state', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resize', type=tuple, default=(512, 384))

    parser.add_argument('--data_dir', type=str, default='D:/stage1/input/data/eval/images')

    args = parser.parse_args()

    data_dir = args.data_dir

    create_dir('submissions')
    inference(data_dir, args)