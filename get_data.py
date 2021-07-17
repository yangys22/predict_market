import os
import time
import datetime
import pandas
import pickle
from typing import List
from config import *
from train_model import *
from tqdm import tqdm


class node:
    def __init__(self, date='', flag=False, files=[]):
        self.date = date
        self.flag = flag
        self.files = files

    def __str__(self):
        return 'date: %s | flag: %s\nfiles: %s' % (self.date, str(self.flag), str(self.files))


def get_files_path(dirpath) -> List[str]:
    print('path: %s' % dirpath)
    print('get files path...')
    time.sleep(0.5)
    file_names = []
    for filepath, dirpath, filenames in tqdm(os.walk(dirpath)):
        if filenames:
            for name in filenames:
                file_name = filepath + '\\' + name
                file_names.append(file_name)
    # 26282
    return file_names


if os.path.exists('model/path'):
    print('load files path...')
    with open('model/path', 'rb') as f:
        files_path_G = pickle.load(f)
else:
    files_path_G = get_files_path(news_data_path)
    with open('model/path', 'wb') as f:
        pickle.dump(files_path_G, f)
print(len(files_path_G))
num_G = 0


def match_date_files(date: str):
    _ = []
    global num_G
    for s in files_path_G:
        if date in s:
            _.append(s)
            num_G += 1
    return _


file_min = 1000
def prepare_data() -> List[node]:
    global file_min
    print('prepare data...')
    data = []
    csv = pandas.read_csv(historical_quotes_csv)
    for line in csv.iloc:
        n = node()
        n.date = str(line[0])
        date_ = datetime.datetime.strptime(n.date, '%Y/%m/%d')
        day = datetime.timedelta(days=1)
        date_ -= day
        n.date = date_.strftime('%Y-%m-%d')
        n.files = match_date_files(n.date)
        if n.files:
            n.flag = (float(line[-1]) > 0)
            data.append(n)
            file_min = min(len(n.files), file_min)
    try:
        with open('model/date_name', 'wb') as f:
            pickle.dump(data, f)
        print('save successfully.')
    except Exception as e:
        print(str(e))
    return data


def get_data():
    print('get x_train and y_train...')
    data = prepare_data()
    print('use files num: %d' % num_G)  # 3566455
    print('files num min: %d' % file_min)  # 1
    from gensim.models.doc2vec import Doc2Vec
    model_dm = Doc2Vec.load("model/1.model")
    train_x = []
    train_y = []
    texts = []
    for d in tqdm(data):  # 3872
        text = ''
        for file in d.files:
            if file[-4:] == '.txt':
                try:
                    f = open(file, 'r', encoding='utf-8')
                    text += f.read()
                    f.close()
                except:
                    try:
                        f = open(file, 'r', encoding='gbk')
                        text += f.read()
                        f.close()
                    except:
                        continue
        if text != '':
            texts.append(text)
            train_y.append(d.flag)
            if len(texts) > 50:
                train_x.extend(toVec(texts, model_dm))
                texts = []

    train_x.extend(toVec(texts, model_dm))
    return train_x, train_y


def main():
    x, y = get_data()
    with open('train_data/x_train', 'wb') as f:
        pickle.dump(x, f)
    with open('train_data/y_train', 'wb') as f:
        pickle.dump(y, f)


if __name__ == '__main__':
    #main()
    with open('model/date_name', 'rb') as f:
        data = pickle.load(f)
    for i in data[:10]:
        print(i)
