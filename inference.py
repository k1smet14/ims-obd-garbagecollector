import warnings
import pickle
from easydict import EasyDict
import json
import os
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from load_data import *


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def load_test_dataset(args):
    with open(args.labeltype_path, 'rb') as f:
        label_type = pickle.load(f)
    test_data = pd.read_csv(args.test_data_path, delimiter='\t', header=None)

    if args.preprocess_type == 0:
        test_dataset = preprocessing_dataset(test_data, label_type)
    elif args.preprocess_type == 1:
        test_dataset = set_entitytoken_dataset(test_data, label_type)

    return test_dataset

def inference(args, model_ckpt):
    # setting
    warnings.filterwarnings(action='ignore')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})

    if 'bert' in args.model_name.split('-'):
        model = BertForSequenceClassification.from_pretrained(model_ckpt)
    elif 'xlm' in args.model_name.split('-'):
        # tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
        # tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})

        model = XLMRobertaForSequenceClassification.from_pretrained(model_ckpt)

    model.to(device)

    # load test data
    test_data = load_test_dataset(args)
    test_label = test_data['label'].values
    
    # tokenize test data
    test_tokenized = tokenized_dataset(test_data, tokenizer, args)

    # set test dataset
    test_dataset = MyDataset(test_tokenized, test_label, args)

    # set test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size = args.val_batch_size, shuffle=False)
    model.eval()
    output_pred=[]
    output_softlabel = []
    # inference
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            outputs = model(
                input_ids = data['input_ids'].to(device),
                attention_mask = data['attention_mask'].to(device),
                #token_type_ids = data['token_type_ids'].to(device)
            )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            output_softlabel.extend(logits)
            result = np.argmax(logits, axis=-1)
            output_pred.extend(result)
    
    output_pred = np.array(output_pred)
    output_softlabel = np.array(output_softlabel)

    # save submission
    create_dir('./prediction')
    create_dir('./prediction/'+args.save_name)
    output = pd.DataFrame(output_pred, columns=['pred'])
    output.to_csv(f"./prediction/{args.save_name}/{model_ckpt.split('/')[-1]}.csv", index=False)

    save_softlabel = pd.DataFrame(output_softlabel)
    save_softlabel.to_csv(f"./prediction/{args.save_name}/{model_ckpt.split('/')[-1]}_soft.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True) # ex) original_config
    parser.add_argument('--ckpt_name', type=str, required=True) # ex) checkpoint-1000
    ipts = parser.parse_args()
    
    # get config
    args = EasyDict()
    with open(f'./config/{ipts.config_name}.json', 'r') as f:
        args.update(json.load(f))
    
    inference_ckpt = ipts.ckpt_name
    model_ckpt = f"/opt/ml/MyBaseline/results/{args.save_name}/{inference_ckpt}"

    print("* inferencing...")
    inference(args, model_ckpt)
    print("* inference successed!")
