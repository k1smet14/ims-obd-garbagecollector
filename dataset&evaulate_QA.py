import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from transformers import pipeline

if __name__ == '__main__':
    train_data = pd.read_csv('/opt/ml/input/data/train/train.tsv', delimiter='\t', header=None)
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)

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

    
    label2q = {}
    for label, question in zip(list(label_type.keys())[1:], questions):
        label2q[label] = question
    
    qa_data = []
    for i in np.array(train_data):
        if i[8] == '관계_없음':
            continue
        data = []
        data.append(i[2]+label2q[i[8]])
        data.append(i[1])
        data.append(i[5])
        qa_data.append(data)

    qa_df = pd.DataFrame(qa_data, columns=['question', 'sentence', 'label'])
    qa_df.to_csv('/opt/ml/input/data/train/train_QA.tsv', sep='\t', index=False, header=True)
    
    train_data_part = train_data[train_data[8] !="관계_없음"]
    train_data_part.reset_index(inplace=True)
    n = len(qa_df)

    nlp = pipeline("question-answering")

    correct = 0
    for i in tqdm(range(n)):
        result= nlp(question=qa_data[i][0], context=qa_data[i][1])
        # print('    question : ', qa_data[i][0])
        # print('model answer : ', result['answer'])
        # print(' real answer : ', train_data_part[5][i])
        # print()
        if train_data_part[5][i] in result['answer']:
            correct+=1
    print(correct/n) # 24.1
  