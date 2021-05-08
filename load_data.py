import pickle as pickle
from easydict import EasyDict
import json

import pandas as pd
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, XLMRobertaTokenizer
from sklearn.model_selection import train_test_split


def preprocessing_dataset(dataset, label_type):
    labels = []
    for label in dataset[8]:
        if label == 'blind':
            labels.append(100)
        else:
            labels.append(label_type[label])

    preprocessed_dataset = pd.DataFrame(
        {'sentence':dataset[1],
        'entity01':dataset[2],
        'entity02':dataset[5],
        'label':labels})

    return preprocessed_dataset


def set_entitytoken_dataset(dataset, label_type):
    sentences = []
    labels = []

    for data in np.array(dataset):
        if data[8] == 'blind':
            labels.append(100)
        else:
            labels.append(label_type[data[8]]) 

        list_sentence = list(data[1])
        ent1_idx = data[3]
        ent2_idx = data[6]
        if ent1_idx > ent2_idx:
            list_sentence.insert(data[4]+1, '[/ENT]')
            list_sentence.insert(data[3], '[ENT]')
            list_sentence.insert(data[7]+1, '[/ENT]')
            list_sentence.insert(data[6], '[ENT]')
        else:
            list_sentence.insert(data[7]+1, '[/ENT]')
            list_sentence.insert(data[6], '[ENT]')
            list_sentence.insert(data[4]+1, '[/ENT]')
            list_sentence.insert(data[3], '[ENT]')
        sentences.append(''.join(list_sentence))

    preprocessed_dataset = pd.DataFrame(
        {'sentence':sentences,
        'entity01':dataset[2],
        'entity02':dataset[5],
        'label':labels})

    return preprocessed_dataset


def load_data(args):
    with open(args.labeltype_path, 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(args.dataset_path, delimiter='\t', header=None)
    
    labels_cnt = Counter(dataset[8])
    sorted_labels_cnt = sorted(labels_cnt.items(), key=lambda x:x[1], reverse=True)
    sorted_labels = [cls[0] for cls in sorted_labels_cnt]

    use_labels = sorted_labels[:args.num_labels]
    dataset = list(filter(lambda x:x[8] in use_labels, np.array(dataset)))
    dataset = pd.DataFrame(dataset)

    if args.preprocess_type == 0:
        dataset = preprocessing_dataset(dataset, label_type)
    elif args.preprocess_type == 1:
        dataset = set_entitytoken_dataset(dataset, label_type)
    
    return dataset


def split_dataset(dataset, args):
    X = np.array(dataset)[:, :-1]
    y = np.array(dataset['label'].values)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.seed)
    train_dataset = pd.DataFrame({'sentence':X_train[:,0], 'entity01':X_train[:,1], 'entity02':X_train[:,2], 'label':y_train})
    val_dataset = pd.DataFrame({'sentence':X_val[:,0], 'entity01':X_val[:,1], 'entity02':X_val[:,2], 'label':y_val})

    return train_dataset, val_dataset


def tokenized_dataset(data, tokenizer, args):
    concat_entity = []
    if 'bert' in args.model_name.split('-'):
        sep_token = '[SEP]'
    elif 'xlm' in args.model_name.split('-'):
        sep_token = '</s>'
    
    if args.preprocess_type == 0:
        for e01, e02 in zip(data['entity01'], data['entity02']):
            temp = e01 + sep_token + e02
            concat_entity.append(temp)
    elif args.preprocess_type == 1:
        for e01, e02 in zip(data['entity01'], data['entity02']):
            temp = '[ENT]' + e01 + '[/ENT]' + sep_token + '[ENT]' + e02 + '[/ENT]'
            concat_entity.append(temp)
    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(data['sentence']),
        return_tensors="pt",
        padding=True,
        truncation='only_second',
        max_length=args.tokenize_maxlen,
        add_special_tokens=True
    )
    return tokenized_sentences


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data, labels, args):
        super(MyDataset, self).__init__()
        self.tokenized_data = tokenized_data
        self.labels = labels
        self.preprocess_type = args.preprocess_type
        self.maxlen = args.tokenize_maxlen

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.tokenized_data.items()}
        
        # if self.preprocess_type == 1: 
        #     entity_vec = torch.zeros(self.maxlen, dtype=int)
        #     start_idx = 0
        #     pass_entity = 0
        #     for idx, token in enumerate(self.tokenized_data['input_ids'][index]):
        #         if token == 119547 and pass_entity > 1:    # [ENT] : 119547
        #             start_idx = idx
        #         elif token == 119548:  # [\ENT] : 119548
        #             if pass_entity > 1:
        #                 entity_vec[start_idx+1:idx] = 1
        #             pass_entity += 1
    
        #     item['token_type_ids'] = item['token_type_ids'] + entity_vec

        item['labels'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)





if __name__ == '__main__':
    args = EasyDict()
    with open('/opt/ml/MyBaseline/config/xlm_config01.json', 'r') as f:
        args.update(json.load(f))

    # check train dataset
    pd.set_option("max_colwidth", 10)


    # check dataset function and sentence

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(len(tokenizer))
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    print(len(tokenizer))
    train_data = load_data(args)
    train_label = train_data['label'].values

    # set train dataset
    train_tokenized = tokenized_dataset(train_data, tokenizer, args)
    print(train_tokenized['input_ids'][0])
    # print(train_tokenized['token_type_ids'][0])
    print(train_tokenized['attention_mask'][0])

    print(tokenizer.tokenize(train_data['sentence'][0]+'</s>'))
    print(tokenizer.decode(train_tokenized['input_ids'][0]))

    train_dataset = MyDataset(train_tokenized, train_label, args)
    # print(train_dataset[0])
    print(train_dataset[0])


'''
    # check tokenizing
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    
    concat_entity = []
    for e01, e02 in zip(dataset01['entity01'], dataset01['entity02']):
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    
    print(tokenizer.tokenize(concat_entity[0], list(dataset01['sentence'])[0]))

    # check max length of tokenized vector
    # len_list = []
    # for ent, sen in zip(concat_entity, list(dataset01['sentence'])):
    #     tokenized_sen = tokenizer.tokenize(ent, sen)
    #     len_list.append(len(tokenized_sen))
     
    # print(max(len_list), np.mean(len_list)) # 328 / 75.045

    # check tokenized_dataset function
    tokenized_data = tokenized_dataset(dataset01, tokenizer, 1)
    print(tokenizer.decode(tokenized_data['input_ids'][0]))
    print(tokenized_data['input_ids'][0])
    print(tokenized_data['token_type_ids'][0])
    print(tokenized_data['attention_mask'][0])

    train_dataset = MyDataset(tokenized_data, dataset01['label'], 1, 100)
    print(train_dataset[0])
'''
