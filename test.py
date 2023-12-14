import os
import time

from sklearn.svm import LinearSVC
import jieba
import thulac

from snownlp import SnowNLP
import joblib
import re


def is_all_Chinese(string):
    for char in string:
        if char > '\u9fa5' or char < '\u4e00':
            return False
    return True


thu1 = thulac.thulac(seg_only=True)


def jieba_cut_sentence(str):
    word_list = []
    str = re.sub('[^\u4e00-\u9fa5]+', '', str)

    words = jieba.cut(str)

    for word in words:
        if is_all_Chinese(word):
            word_list.append(word)
    return word_list


def snow_cut_sentence(str):
    word_list = []
    str = re.sub('[^\u4e00-\u9fa5]+', '', str)

    words = SnowNLP(str).words

    for word in words:
        if is_all_Chinese(word):
            word_list.append(word)
    return word_list


def thu_cut_sentence(str):
    word_list = []
    str = re.sub('[^\u4e00-\u9fa5]+', '', str)

    words = thu1.cut(str, text=True).split(" ")

    for word in words:
        if is_all_Chinese(word):
            word_list.append(word)
    return word_list


count = 0


def read_mail(path, word_set, func):
    global count
    count += 1
    sentences = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line[0] > '\u9fa5' or line[0] < '\u4e00':
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
            if line == '' or line[0] > '\u9fa5' or line[0] < '\u4e00':
                continue
            sentences.append(line)
        f.close()
    for line in sentences:
        for word in func(line):
            vector[word + '\n'] = 1 if is_all_Chinese(word) else 0
    return vector


def write_full_words(func):
    word_set = {}
    root_path = './trec06c/data'
    sequences = os.listdir(root_path)
    sequences.sort()
    sequences = sequences[:20]
    for dir in sequences:
        files = os.listdir(os.path.join(root_path, dir))
        files.sort()
        for file in files:
            read_mail(os.path.join(root_path, dir, file), word_set, func)
    with open('full_words.txt', 'w') as file:
        for word in word_set.keys():
            file.write(f'%s\n' % word)
    file.close()


def get_full_words() -> dict:
    word_set = {}
    file = open('full_words.txt', 'r')
    for line in file.readlines():
        word_set[line] = 0
    return word_set


def get_label():
    list = []
    with open('./trec06c/full/index', 'r') as file:
        lines = file.readlines()
        for line in lines:
            list.append(line.split(' ')[0])
    return list


if __name__ == '__main__':

    func = snow_cut_sentence

    start_dict_time = time.time()
    write_full_words(func)
    end_dict_time = time.time()

    labels = get_label()
    datas = []
    word_set = get_full_words()
    root_path = './trec06c/data'
    sequences = os.listdir(root_path)
    sequences.sort()
    sequences = sequences[:20]
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

    svc = LinearSVC(dual=False)
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
    svc.fit(train_data, train_label)
    joblib.dump(svc, 'model.pkl')
    print(svc.score(test_data, test_label))
    print(end_dict_time - start_dict_time)
    print(end_partition_time - start_partition_time)
