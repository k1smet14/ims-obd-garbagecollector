import warnings
import gc

import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering, Trainer, TrainingArguments

def killmemory():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, start_idxs, end_idxs):
        super(MyDataset, self).__init__()
        self.tokenized_data = tokenized_data
        self.start_idxs = start_idxs
        self.end_idxs = end_idxs

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.tokenized_data.items()}
        item['start_positions'] = torch.tensor(self.start_idxs[index])
        item['end_positions'] = torch.tensor(self.end_idxs[index])
        return item

    def __len__(self):
        return len(self.start_idxs)


if __name__=='__main__':
    # setting
    killmemory()
    seed_everything(7)
    warnings.filterwarnings(action='ignore')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset_path = "/opt/ml/input/data/train/train_QA.tsv"
    train_data = pd.read_csv(dataset_path, delimiter='\t')

    questions = list(train_data['question'])
    texts = list(train_data['sentence'])
    labels = list(train_data['label'])

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_data = tokenizer(texts,
        questions,
        return_tensors="pt",
        padding=True,
        truncation="only_first",
        max_length=100,
        add_special_tokens=True)
    
    tokenized_label = tokenizer(labels, add_special_tokens=False)['input_ids']

    start_idxs = []
    end_idxs = []
    for data, label in zip(tokenized_data['input_ids'], tokenized_label):
        data = data.cpu().numpy()
        label = np.array(label)
        start_idx = 0
        end_idx = 0
        correct = 0
        
        start_idx_list = np.where(data==label[0])[0]
        for idx in start_idx_list:
            label_in_text = data[idx:idx+len(label)]
            if list(label_in_text) == list(label):
                start_idx = idx
                end_idx = idx+len(label)-1
        

        start_idxs.append(start_idx)
        end_idxs.append(end_idx)
    

    train_dataset = MyDataset(tokenized_data, start_idxs, end_idxs)
    
    # print(tokenized_data['input_ids'][0])
    # print(tokenized_data['token_type_ids'][0])
    # print(tokenized_data['attention_mask'][0])
    # print(tokenizer.decode(tokenized_data['input_ids'][0]))

    # model
    model_config = BertConfig.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=model_config)
    model.to(device)

    # training
    training_args = TrainingArguments(
        output_dir='./results/useQA',
        save_total_limit=5,
        save_steps=500,
        num_train_epochs=10,
        learning_rate=5e-5,               
        per_device_train_batch_size=16,  
        warmup_steps=300,              
        weight_decay=0.01,           
        logging_dir='./logs',         
        logging_steps=100,                      
        label_smoothing_factor=0.5
    )

    trainer = Trainer(
        model=model,                        
        args=training_args,                 
        train_dataset=train_dataset
    )

    trainer.train()

