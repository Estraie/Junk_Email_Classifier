import os
import time
import argparse
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import rfflearn.cpu as rfflearn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

import joblib
import re

import nltk
from nltk.data import find

# 定义分词所需的数据文件
data_file = 'tokenizers/punkt'

# 检查文件是否存在
if find(data_file):
    print(f"{data_file} already exists. No need to download.")
else:
    # 如果文件不存在，下载分词所需的数据
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

import spacy

from transformers import BertTokenizer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default="svc", type=str)
parser.add_argument('--tokenizer', default="nltk", type=str)
parser.add_argument('--folders', default=20, type=int)
parser.add_argument('--rff', default=False, type=bool)
parser.add_argument('--dim', default=1024, type=int)
args = parser.parse_args()

def is_all_English(string):
    for char in string:
        if char.isalpha():
            return False
    return True

### spaccy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
def spaCy(text):
    doc = nlp(text)
    return [token.text for token in doc]
###

### huggingface
tokenizer_hug = BertTokenizer.from_pretrained("bert-base-uncased")
def hug(text):
    tokens = tokenizer_hug.tokenize(text)
    return tokens
###

count = 0


def read_mail(path, word_set, func):
    global count
    count += 1
    sentences = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or not line[0].isalpha():
                continue
            sentences.append(line)
        f.close()
    for line in sentences:
        for word in func(line):
            word_set[word] = 1



def mail2vector(path, word_set: dict, func):
    vector = word_set.copy()
    global count
    count += 1
    sentences = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or not line[0].isalpha():
                continue
            if line.startswith(("Received:","Newsgroups:","Path:","From:","Subject:","Message-Id:","Sender:","Organization:","Date:","Content-Type:","Content-Length:","Apparently-To:")):
                continue
            sentences.append(line)
        f.close()
    for line in sentences:
        for word in func(line):
            vector[word + '\n'] = 1 if is_all_English(word) else 0
    return vector


def write_full_words(func):
    word_set = {}
    root_path = './trec06p/data'
    sequences = os.listdir(root_path)
    sequences.sort()
    sequences = sequences[:args.folders]
    for dir in sequences:
        files = os.listdir(os.path.join(root_path, dir))
        files.sort()
        for file in files:
            read_mail(os.path.join(root_path, dir, file), word_set, func)
    with open('full_words_p.txt', 'w') as file:
        for word in word_set.keys():
            file.write(f'%s\n' % word)
    file.close()


def get_full_words() -> dict:
    word_set = {}
    file = open('full_words_p.txt', 'r')
    for line in file.readlines():
        word_set[line] = 0
    return word_set


def get_label():
    list = []
    with open('./trec06p/full/index', 'r') as file:
        lines = file.readlines()
        for line in lines:
            list.append(line.split(' ')[0])
    return list


if __name__ == '__main__':

    if(args.tokenizer=="nltk"):
        func = word_tokenize
    elif(args.tokenizer=="spacy"):
        func = spaCy
    elif(args.tokenizer=="hug"):
        func = hug
    else:
        func=None

    start_dict_time = time.time()
    write_full_words(func)
    end_dict_time = time.time()

    labels = get_label()
    datas = []
    word_set = get_full_words()
    root_path = './trec06p/data'
    sequences = os.listdir(root_path)
    sequences.sort()
    sequences = sequences[:args.folders]
    file_count = 0

    start_partition_time = time.time()
    for dir in sequences:
        files = os.listdir(os.path.join(root_path, dir))
        files.sort()
        for file in files:
            vector = mail2vector(os.path.join(root_path, dir, file), word_set, func)
            datas.append(list(vector.values()))
            file_count += 1
    end_partition_time = time.time()

    if(args.model=="svc"):
        model = LinearSVC(dual=False)
    elif(args.model=="logi"):
        model = LogisticRegression()
    elif(args.model=="tree"):
        model=DecisionTreeClassifier()
    elif(args.model=="forest"):
        model=RandomForestClassifier()
    elif(args.model=="ensemble"):
        svc = SVC(probability=True)
        logi = LogisticRegression()
        forest = RandomForestClassifier()
        model=VotingClassifier(estimators=[('svc', svc), ('logi', logi), ('forest', forest)], voting='soft')


    labels2 = []
    for label in labels[:len(datas) + 1]:
        if label == 'spam':
            labels2.append(0)
        else:
            labels2.append(1)
    train_data = datas[:int(len(datas) * 0.8)]
    train_label = labels2[:int(len(datas) * 0.8)]
    test_data = datas[int(len(datas) * 0.8):]
    test_label = labels2[int(len(datas) * 0.8): len(datas)]

    print(f"{args.model}_{args.tokenizer}")
    start_fit_time = time.time()
    if(args.rff==True):
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        test_data=np.array(test_data)
        test_label=np.array(test_label)
        model = rfflearn.RFFSVC(dim_kernel=args.dim).fit(train_data, train_label) 
        # svc.fit(train_data, train_label)
        joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
        y_preds=model.predict(test_data)
        train_preds=model.predict(train_data)
        print(f"train:{accuracy_score(train_label,train_preds)}")
        print(f"test:{accuracy_score(test_label,y_preds)}")
    else:
        if(args.model=="svc"):
            # train_data=np.array(train_data)
            # train_label=np.array(train_label)
            # test_data=np.array(test_data)
            # test_label=np.array(test_label)
            # svc = rfflearn.RFFSVC(dim_kernel=4096).fit(train_data, train_label) 
            model.fit(train_data, train_label)
            joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
            y_preds=model.predict(test_data)
            train_preds=model.predict(train_data)
            print(f"train:{accuracy_score(train_label,train_preds)}")
            print(f"test:{accuracy_score(test_label,y_preds)}")
        elif(args.model=="logi"):
            model.fit(train_data, train_label)
            joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
            y_preds=model.predict(test_data)
            train_preds=model.predict(train_data)
            print(f"train:{accuracy_score(train_label,train_preds)}")
            print(f"test:{accuracy_score(test_label,y_preds)}")
        elif(args.model=="tree"):
            model.fit(train_data, train_label)
            joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
            y_preds=model.predict(test_data)
            train_preds=model.predict(train_data)
            print(f"train:{accuracy_score(train_label,train_preds)}")
            print(f"test:{accuracy_score(test_label,y_preds)}")
        elif(args.model=="forest"):
            model.fit(train_data, train_label)
            joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
            y_preds=model.predict(test_data)
            train_preds=model.predict(train_data)
            print(f"train:{accuracy_score(train_label,train_preds)}")
            print(f"test:{accuracy_score(test_label,y_preds)}")
        elif(args.model=="ensemble"):
            model.fit(train_data, train_label)
            joblib.dump(model, f'{args.model}_{args.tokenizer}_model_p.pkl')
            y_preds=model.predict(test_data)
            train_preds=model.predict(train_data)
            print(f"train:{accuracy_score(train_label,train_preds)}")
            print(f"test:{accuracy_score(test_label,y_preds)}")
    end_fit_time=time.time()

    # 计算混淆矩阵
    matrix = confusion_matrix(test_label, y_preds)

    # 使用 seaborn 画热图
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')

    # 设置图形标题和轴标签（可根据需要调整）
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # 保存图形到文件（可根据需要调整文件名和格式）
    plt.savefig(f'confusion_matrix_en{args.model}.png')

    inference_start_time=time.time()
    model.predict(np.array([test_data[-1]]))
    inference_end_time=time.time()

    print(f"dict time{end_dict_time - start_dict_time}")
    print(f"partition time{end_partition_time - start_partition_time}")
    print(f"fit time{end_fit_time-start_fit_time}")
    print(f"fit time{inference_end_time-inference_start_time}")