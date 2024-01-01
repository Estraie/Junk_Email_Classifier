import os
import time

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import jieba

import joblib
import re


def is_all_Chinese(string):
    for char in string:
        if char > '\u9fa5' or char < '\u4e00':
            return False
    return True


#thu1 = thulac.thulac(seg_only=True)


def jieba_cut_sentence(str):
    word_list = []
    str = re.sub('[^\u4e00-\u9fa5]+', '', str)

    words = jieba.cut(str)

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

    func = jieba_cut_sentence

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
    
    print(end_dict_time - start_dict_time)
    print(end_partition_time - start_partition_time)
    
    t1 = time.time()
    # SVC
    svc = LinearSVC(dual=False)
    svc.fit(train_data, train_label)
    joblib.dump(svc, 'svc_model.pkl')
    t1_2 = time.time()
    
    train_accuracy = svc.score(train_data, train_label)
    test_accuracy = svc.score(test_data, test_label)
    print("SVC Train/Test Accuracy", train_accuracy, test_accuracy)
    print("SVC time cost:", t1_2 - t1)
    
    y_preds = svc.predict(test_data)
    matrix = confusion_matrix(test_label, y_preds)
    plt.figure(dpi = 160)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of SVC Result')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix_svc_cn.png')
    plt.clf()
    
    t2 = time.time()
    
    # Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_data, train_label)
    joblib.dump(rf_classifier, 'random_forest_model.pkl')
    t2_2 = time.time()
    
    train_accuracy = rf_classifier.score(train_data, train_label)
    test_accuracy = rf_classifier.score(test_data, test_label)
    print("Random Forest Train/Test Accuracy:", train_accuracy, test_accuracy)
    print("Random Forest time cost:", t2_2 - t2)
    
    y_preds = rf_classifier.predict(test_data)
    matrix = confusion_matrix(test_label, y_preds)
    plt.figure(dpi = 160)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of Random Forest Result')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix_rf_cn.png')
    plt.clf()
    
    t3 = time.time()
    
    # Logistic
    log_classifier = LogisticRegression()
    log_classifier.fit(train_data, train_label)
    joblib.dump(log_classifier, 'logistic_model.pkl')
    
    t3_2 = time.time()
    
    train_accuracy = log_classifier.score(train_data, train_label)
    test_accuracy = log_classifier.score(test_data, test_label)
    print("Logistic Regression Train/Test Accuracy:", train_accuracy, test_accuracy)
    print("Logistic Regression time cost:", t3_2 - t3)
    
    y_preds = log_classifier.predict(test_data)
    matrix = confusion_matrix(test_label, y_preds)
    plt.figure(dpi = 160)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of Logistic Regression Result')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix_lg_cn.png')
    plt.clf()
    
    t4 = time.time()
    
    # Decision Tree
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(train_data, train_label)
    joblib.dump(dt_classifier, 'decision_tree.pkl')
    
    t4_2 = time.time()
    
    train_accuracy = dt_classifier.score(train_data, train_label)
    test_accuracy = dt_classifier.score(test_data, test_label)
    print("Decision Tree Train/Test Accuracy:", train_accuracy, test_accuracy)
    print("Decision Tree time cost:", t4_2 - t4)
    
    y_preds = dt_classifier.predict(test_data)
    matrix = confusion_matrix(test_label, y_preds)
    plt.figure(dpi = 160)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of Decision Tree Result')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix_dt_cn.png')
    plt.clf()