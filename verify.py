import joblib
from test import get_full_words, mail2vector, jieba_cut_sentence, is_all_Chinese, thu_cut_sentence, snow_cut_sentence
import os


def mail2vector_test(path, word_set: dict, func):
    vector = word_set.copy()
    sentences = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            line = line.strip()
            if line == '' or line[0] > '\u9fa5' or line[0] < '\u4e00':
                continue
            sentences.append(line)
        f.close()
    for line in sentences:
        for word in func(line):
            if vector.get(word + '\n') is None:
                continue
            vector[word + '\n'] = 1 if is_all_Chinese(word) else 0
    return vector


svc = joblib.load('model.pkl')

if __name__ == '__main__':
    func = snow_cut_sentence
    datas = []
    word_set = get_full_words()
    test_file = os.listdir('test_data')
    test_file.sort()
    for file in test_file:
        vector = mail2vector_test(os.path.join('test_data', file), word_set, func)
        feature = list(vector.values())
        datas.append(feature)
    result = svc.predict(datas)
    spam = 'a junk mail'
    good = 'a normal mail'
    for i in range(len(test_file)):
        print(f'{test_file[i]} is {spam if result[i] == 0 else good}')
