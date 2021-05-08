from easydict import EasyDict
import json
import wandb
import gc
import warnings
import argparse

import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments, \
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer

from sklearn.metrics import accuracy_score

from load_data import *


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def killmemory():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train(args):

    # setting 
    killmemory()
    seed_everything(args.seed)
    warnings.filterwarnings(action='ignore')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.preprocess_type == 1:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})

    # set model hyperparameter
    if 'bert' in args.model_name.split('-'):
        print('* bert is choosen')
        model_config = BertConfig.from_pretrained(args.model_name)
        model_config.num_labels = args.num_labels
        model = BertForSequenceClassification.from_pretrained(args.model_name, config=model_config)
        # if args.preprocess_type == 1:
        #     model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(3, 768)

    elif 'xlm' in args.model_name.split('-'):
        print('* xlm is choosen')
        model_config = XLMRobertaConfig.from_pretrained(args.model_name)
        model_config.num_labels = args.num_labels
        model = XLMRobertaForSequenceClassification.from_pretrained(args.model_name, config=model_config)
    
    print('* len tokenizer : ', len(tokenizer))
    if args.preprocess_type == 1:
        model.resize_token_embeddings(len(tokenizer))
    print('* model')
    # print(model.bert.embeddings)
    model.to(device)
    

    if not args.train_val_split: 
        print("* do not split train, val set")
        # load data
        train_data = load_data(args)
        train_label = train_data['label'].values

        # set train dataset
        train_tokenized = tokenized_dataset(train_data, tokenizer, args)
        train_dataset = MyDataset(train_tokenized, train_label, args)

        create_dir('./results/'+args.save_name)
        # set training options
        training_args = TrainingArguments(
            output_dir='./results/'+args.save_name,          # output directory
            save_total_limit=args.save_total_limit,              # number of total save model.
            save_steps=args.save_step,                 # model saving step.
            num_train_epochs=args.epoch,              # total number of training epochs
            learning_rate=args.lr,               # learning_rate
            per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
            warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,              # log saving step.
            label_smoothing_factor=0.5
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            compute_metrics=compute_metrics         # define metrics function
        )

    elif args.train_val_split: 
        print("* split train, val set")
        # load data
        data = load_data(args)
        train_data, val_data = split_dataset(data, args)

        train_label = train_data['label'].values
        val_label = val_data['label'].values

        # set train dataset
        train_tokenized = tokenized_dataset(train_data, tokenizer, args)
        train_dataset = MyDataset(train_tokenized, train_label, args)
        val_tokenized = tokenized_dataset(val_data, tokenizer, args)
        val_dataset = MyDataset(val_tokenized, val_label, args)
        
        create_dir('./results')
        create_dir('./results/'+args.save_name)
        # set training options
        training_args = TrainingArguments(
            output_dir='./results/'+args.save_name,          # output directory
            save_total_limit=args.save_total_limit,              # number of total save model.
            save_steps=args.save_step,                 # model saving step.
            num_train_epochs=args.epoch,              # total number of training epochs
            learning_rate=args.lr,               # learning_rate
            per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.val_batch_size,   # batch size for evaluation
            warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,              # log saving step.
            evaluation_strategy='steps', # evaluation strategy to adopt during training
            eval_steps = args.save_step,            # evaluation step.
            run_name = args.save_name,
            label_smoothing_factor=args.label_smoothing_factor


            # report_to = 'wandb',        # enable logging to W&B
            # run_name = args.model_name
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics         # define metrics function
        )

    # train model
    trainer.train()



if __name__=='__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True) # ex) original_config
    ipts = parser.parse_args()

    # get config
    args = EasyDict()
    with open(f'./config/{ipts.config_name}.json', 'r') as f:
        args.update(json.load(f))

    # # wandb initializing
    # wandb.init(project='boostcamp_stage2', entity='doooom')
    # wandb.config.update(args)
    os.environ['WANDB_PROJECT'] = 'boostcamp_stage2'
    os.environ['WANDB_LOG_MODEL'] = 'true'

    # training
    print("* training...")
    train(args)
    print("* train successed!")

    