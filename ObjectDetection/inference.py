import json
import argparse
from easydict import EasyDict
from importlib import import_module

from tqdm import tqdm
import os
import warnings
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


from torch.utils.data import DataLoader
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
def collate_fn(batch):
    return tuple(zip(*batch))

def inference_fn(test_dataloader, model, device):
    model.eval()
    outputs = []
    for images, targets, image_ids in tqdm(test_dataloader):

        images = list(image.float().to(device) for image in images)
        output = model(images)

        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'labels': out['labels'].tolist(), 'scores': out['scores'].tolist()})
    
    return outputs


def inference(ipts, args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_augmentation_module = getattr(import_module("augmentation"), 'ValidAugmentation')
    test_augmentation = test_augmentation_module(resize=args.resize) 

    dataset_module = getattr(import_module("dataset"), args.dataset)
    test_dataset = dataset_module(args.test_annotation, args.data_dir, test_augmentation)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )  

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes = 11, args=args) 
    check_point = f'./saved_model/({ipts.config}){args.config_name}_17.pth'
    model.load_state_dict(torch.load(check_point))
    model.to(device)

    outputs = inference_fn(test_dataloader, model, device)

    prediction_strings = []
    file_names = []
    coco = COCO(args.test_annotation)
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > args.test_score_threshold:
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0]/(args.resize[0]/512)) + ' ' + str(
                    box[1]/(args.resize[0]/512)) + ' ' + str(box[2]/(args.resize[0]/512)) + ' ' + str(box[3]/(args.resize[0]/512)) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'./submissions/({ipts.config}){args.config_name}.csv', index=None)


if __name__ == '__main__':
    create_dir('submissions')
    warnings.filterwarnings(action='ignore')

    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get args in config file
    args = EasyDict()
    with open(f'./config/{ipts.config}.json', 'r') as f:
        args.update(json.load(f))
    
    inference(ipts, args)
