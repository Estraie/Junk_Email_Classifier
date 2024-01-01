import joblib
from train_cn import get_full_words, mail2vector, jieba_cut_sentence, is_all_Chinese
import os


def mail2vector_test(path, word_set: dict, func):
    vector = word_set.copy()
    sentences = []
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
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


svc = joblib.load('svc_model.pkl')
rf = joblib.load('random_forest_model.pkl')
dt = joblib.load('decision_tree.pkl')
lg = joblib.load('logistic_model.pkl')

if __name__ == '__main__':
    func = jieba_cut_sentence
    datas = []
    word_set = get_full_words()
    test_file = os.listdir('test_data_cn')
    test_file.sort()
    for file in test_file:
        vector = mail2vector_test(os.path.join('test_data_cn', file), word_set, func)
        feature = list(vector.values())
        datas.append(feature)
    result_svc = svc.predict(datas)
    result_rf = rf.predict(datas)
    result_dt = dt.predict(datas)
    result_lg = lg.predict(datas)
    spam = 'a junk mail'
    good = 'a normal mail'
    
    print('SVC:')
    for i in range(len(test_file)):
        print(f'{test_file[i]} is {spam if result_svc[i] == 0 else good}')
        
    print('Random Forest:')
    for i in range(len(test_file)):    
        print(f'{test_file[i]} is {spam if result_rf[i] == 0 else good}')
        
    print('Logistic:')
    for i in range(len(test_file)):
        print(f'{test_file[i]} is {spam if result_lg[i] == 0 else good}')
        
    print('Decision Tree:')
    for i in range(len(test_file)):
        print(f'{test_file[i]} is {spam if result_dt[i] == 0 else good}')
