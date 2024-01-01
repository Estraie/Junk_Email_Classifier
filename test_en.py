import joblib
from nltk.tokenize import word_tokenize
from Junk_Email_Classifier.train_en import get_full_words, mail2vector, is_all_English
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default="svc", type=str)
parser.add_argument('--tokenizer', default="nltk", type=str)
args = parser.parse_args()

def mail2vector_test(path, word_set: dict, func):
    vector = word_set.copy()
    sentences = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        print(lines)
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
            if vector.get(word + '\n') is None:
                continue
            vector[word + '\n'] = 1 if is_all_English(word) else 0
    return vector


svc = joblib.load(f'{args.model}_{args.tokenizer}_model_p.pkl')

if __name__ == '__main__':
    func = word_tokenize
    datas = []
    word_set = get_full_words()
    test_file = os.listdir('test_data_en')
    test_file.sort()
    for file in test_file:
        vector = mail2vector_test(os.path.join('test_data_en', file), word_set, func)
        feature = list(vector.values())
        datas.append(feature)
    # datas=np.array(datas)
    print(len(datas[0]))
    result = svc.predict(np.array(datas))
    spam = 'a junk mail'
    good = 'a normal mail'
    for i in range(len(test_file)):
        print(f'{test_file[i]} is {spam if result[i] == 0 else good}')
