import pickle

import numpy
import xgboost

import get_data
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm
# get_data.main()
print('load train data...')
with open('train_data/x_train', 'rb') as f:
    x = pickle.load(f)
with open('train_data/y_train', 'rb') as f:
    y = pickle.load(f)

x = numpy.array(x)
y = numpy.array(y)
# print(x)
print('length x: %d, length y: %d' % (len(x), len(y)))
time.sleep(0.5)
# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)

# 定义各类分类算法
lda = LinearDiscriminantAnalysis()
svc = svm.SVC()  # 支持向量机，SVM
mlp = MLPClassifier(max_iter=1000)  # 多层感知机，MLP
dtc = DecisionTreeClassifier()  # 决策树,DT
knc = KNeighborsClassifier()  # K最近邻,KNN
bnb = BernoulliNB()  # 伯努利贝叶斯，BNB
gnb = GaussianNB()  # 高斯贝叶斯,GNB
lgr = LogisticRegression()  # 逻辑回归，LGR
rfc = RandomForestClassifier()  # 随机森林，RFC
abc = AdaBoostClassifier()  # AdaBoost
gbm = lightgbm.LGBMClassifier()  # lightgbm

models = [lda,svc,mlp,dtc,knc,bnb,gnb,lgr,rfc,abc,gbm]
# # 神经元个数
# units = []
# for i in range(1, 30):
#     for j in range(1, 30):
#         #for k in range(1, 30):
#         units.append([i, j])
# a, b = [], 0

    # 激活函数：relu, logistic, tanh
    # 优化算法：lbfgs, sgd, adam。adam适用于较大的数据集，lbfgs适用于较小的数据集。
    # 初始化模型
    # ann_model = MLPClassifier(hidden_layer_sizes=unit, activation='relu', solver='adam', random_state=0,
    #                           max_iter=1000)
    # 训练模型
for model in models:
    print(model, end='\t')
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print('准确率：{:.3f}'.format(score))
    # if score > b:
    #     a, b = unit, score
    # joblib.dump(ann_model, 'model/mlp.model')
    # print('模型保存完成.')


# with open('log.txt', 'a+') as f:
#     f.write(str(a) + '\t' + str(b) + '\n')
