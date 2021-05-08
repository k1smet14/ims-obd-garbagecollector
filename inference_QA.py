import warnings
from tqdm import tqdm

import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering, Trainer, TrainingArguments

from train_QA import MyDataset

questions = ['의 배우자는 누구인가요?',
        '의 직업은 무엇인가요?',
        '의 모회사는 무엇인가요?',
        '의 소속 단체는 무엇인가요?',
        '의 동료는 누구인가요?',
        '의 별칭은 무엇인가요?',
        '의 국적은 어디인가요?',
        '의 부모님은 누구인가요?',
        '의 본사 국가는 어디인가요?',
        '의 구성원은 누구인가요?',
        '의 친족은 누구인가요?',
        '의 창립자는 누구인가요?',
        '의 주주는 누구인가요?',
        '의 사망 일시는 언제인가요?',
        '의 상위 단체는 무엇인가요?',
        '의 본사는 어디인가요?',
        '는 무엇을 제작하나요?',
        '의 사망 원인은 무엇인가요?',
        '의 출생 도시는 어디인가요?',
        '의 본사 도시는 어디인가요?',
        '의 자녀는 누구인가요?',
        '는 무엇을 제작하였나요?',
        '의 하위 단체는 무엇인가요?',
        '의 별칭은 무엇인가요?',
        '의 형제, 자매, 남매는 누구인가요?',
        '의 출생 국가는 어디인가요?',
        '의 출생 일시는 어디인가요?',
        '의 구성원 수는 몇 명인가요?',
        '의 자회사는 어디인가요?',
        '는 어디에 거주하나요?',
        '의 해산일은 언제인가요?',
        '의 거주 도시는 어디인가요?',
        '의 창립일은 언제인가요?',
        '의 종교는 무엇인가요?',
        '의 거주 국가는 어디인가요?',
        '는 무엇의 용의자인가요?',
        '의 사망 도시는 어디인가요?',
        '의 정치, 종교 성향은 무엇인가요?',
        '의 학교는 어디인가요?',
        '의 사망 국가는 어디인가요?',
        '의 나이는 몇 살인가요?']


if __name__=='__main__':
    # setting
    warnings.filterwarnings(action='ignore')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_path = "/opt/ml/input/data/test/test.tsv"
    test_data = pd.read_csv(dataset_path, delimiter='\t', header=None)

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_ckpt = "/opt/ml/MyBaseline/results/useQA/checkpoint-2000"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()

    for data in tqdm(np.array(test_data)):
        texts = [data[1]] * len(questions)
        questions = [data[2] + question for question in questions]
        labels = [data[5]] * len(questions)

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
            break
        start_idxs = start_idxs * len(questions)
        end_idxs = end_idxs * len(questions)
        
        test_dataset = MyDataset(tokenized_data, start_idxs, end_idxs)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

        with torch.no_grad():
            for data in tqdm(test_dataloader):
                outputs = model(
                    input_ids = data['input_ids'].to(device),
                    attention_mask = data['attention_mask'].to(device),
                    token_type_ids = data['token_type_ids'].to(device)
                )
                start_idx = outputs[0]
                start_idx = start_idx.detach().cpu().numpy()
                start_idx = np.argmax(start_idx)

                end_idx = outputs[1]
                end_idx = end_idx.detach().cpu().numpy()
                end_idx = np.argmax(end_idx)
                print(start_idx, end_idx)

                result_token = data['input_ids'].detach().cpu().numpy().flatten()[start_idx:end_idx+1]
                print(result_token)    
                result = tokenizer.decode(result_token)
                     
                print(result)  

        break
