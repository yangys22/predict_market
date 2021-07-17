import multiprocessing
import string
import time

from typing import List

import get_data
import jieba
import os
import gensim
import datetime
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
for s in string.punctuation:
    stop_words.append(s)


def get_datasest(file_names: list):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument

    x_train = []
    error = 0
    for i, file_name in enumerate(tqdm(file_names)):
        try:
            with open(file_name, 'r', encoding='utf-8') as cf:
                text = cf.read()
                for s in stop_words:
                    while s in text:
                        text = text.replace(s, '')
                try:
                    word_list = text.split(' ')

                    while ' ' in word_list:
                        word_list.remove(' ')
                    while '\n' in word_list:
                        word_list.remove('\n')

                    document = TaggededDocument(word_list, tags=[i])
                    x_train.append(document)
                except:
                    pass
        except:
            error+=1
    print(error)
    return x_train


def train(x_train, size=100):  # size是最终训练出的句子向量的维度
    if os.path.exists('model/1.model'):

        print()
        print('加载模型...')
        global n
        print(os.path.getsize('model/1.model')-n)
        n = os.path.getsize('model/1.model')
        s = datetime.datetime.now()
        model_dm = Doc2Vec.load("model/1.model")
        print('用时 %s' % str(datetime.datetime.now() - s))

        print()
        print('开始训练...')
        s= datetime.datetime.now()
        model_dm.train(x_train, epochs=20, total_examples=model_dm.corpus_count)
        print('用时 %s' % str(datetime.datetime.now() - s))

        print()
        print('start save...')
        s = datetime.datetime.now()
        model_dm.save('model/1.model')  # 模型保存的位置
        print('用时 %s' % str(datetime.datetime.now() - s))
    else:

        print('start train...')
        s = datetime.datetime.now()
        model_dm = Doc2Vec(x_train, workers=multiprocessing.cpu_count(), min_count=1, window=5, negative=10, sample=1e-5, cbow_mean=0, epochs=20)

        print('用时 %s' % str(datetime.datetime.now() - s))
        # print('...')
        # print(datetime.datetime.now())
        # model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)

        print()
        print('start save...')
        s = datetime.datetime.now()
        model_dm.save('model/1.model')  # 模型保存的位置

        print('用时 %s' % str(datetime.datetime.now()-s))


def toVec(texts: List[str], model_dm):
    # TaggededDocument = gensim.models.doc2vec.TaggedDocument
    # x_train = []
    # for i, text in enumerate(ss):
    #     word_list = ' '.join(jieba.cut(text.split('\n')[0]))
    #     word_list = word_list.split(' ')
    #     l = len(word_list)
    #     word_list[l - 1] = word_list[l - 1].strip()
    #     document = TaggededDocument(word_list, tags=[i])
    #     x_train.append(document)
    vectors = []
    print('\nto vec...')
    # model_dm.train(x_train, epochs=20, total_examples=model_dm.corpus_count)
    for text in texts:
        for s in stop_words:
            while s in text:
                text = text.replace(s, '')
        test_text = text.split(' ')
        while ' ' in test_text:
            test_text.remove(' ')
        while '\n' in test_text:
            test_text.remove('\n')
        inferred_vector_dm = model_dm.infer_vector(test_text)  ##得到文本的向量

        vectors.append(inferred_vector_dm)


    # model_dm.save('model/1.model')  # 模型保存的位置

    return vectors


n = 0
if __name__ == '__main__':
    """
    for i in range(19):
        print()
        print('第%d个...' % i)
        s = datetime.datetime.now()
        train(get_datasest('m/'+str(i)+'.txt'))
        print()
        print('总用时 %s' % str(datetime.datetime.now()-s))

    """
    s = datetime.datetime.now()
    x = get_datasest(get_data.files_path_G)
    train(x)
    print()
    print('总用时 %s' % str(datetime.datetime.now() - s))
    # l = '总用时 %s % str(datetime.datetime.now() - s'
    # t = toVec(l)
    # print(t)
