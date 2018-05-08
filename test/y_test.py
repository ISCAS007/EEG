# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas

data_path = '/media/sdb/xiaobingdu/data_bmovie2.mat'
# load dataset
datamat = sio.loadmat(data_path)
data = datamat['xlsdata']
print(data.shape)

y_col = 151
x = data[:, :150]
y_raw = data[:, y_col]

labels = np.unique(y_raw)
y = np.zeros_like(y_raw)

results = {}
for label in labels:
    if y_col == 151:
        y[y_raw == label] = 0
        y[y_raw != label] = 1

    print('label is', label)
    print('unique is', np.unique(y))

    use_ratio = 0.1
    x_use, x_unuse, y_use, y_unuse = train_test_split(
        x, y, test_size=1-use_ratio)

    x_train, x_test, y_train, y_test = train_test_split(
        x_use, y_use, test_size=0.33)
    scaler = preprocessing.StandardScaler().fit(x_use)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print('x_train shape is', x_train.shape)
    print('y_train shape is', y_train.shape)

    clf = svm.SVC(C=1.0)
    # clf=DecisionTreeClassifier(max_depth=15)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)

    print('train acc', train_acc)
    print('test acc', test_acc)

    results[str(label)] = [train_acc, test_acc]
    
print(results)
a=pandas.DataFrame()
b=a.from_dict(results)
b.index=['train','test']
print(b)
