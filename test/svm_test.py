# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.linear_model import SGDClassifier

data_path='/media/sdb/xiaobingdu/data_bmovie2.mat'
#load dataset
datamat = sio.loadmat(data_path)
data = datamat['xlsdata']
print(data.shape)

y_col=151
x=data[:,:150]
y=data[:,y_col]
#x=x[y!=102,:]
#y=y[y!=102]
if y_col == 151:
    y[np.logical_or(y==103,y==104)]=102
    y[np.logical_or(y==106,y==107)]=105
    y[y==109]=108
    y[y==112]=111
    y[y==114]=113
    y[y==116]=115
    
#for i in range(datarow):
#    if data[i, 151:152] == 101:
#        newlabel[i, 0] = 0;
#    if data[i, 151:152] == 102 or data[i, 151:152] == 103 or data[i, 151:152] == 104:
#        newlabel[i, 0] = 1;
#    if data[i, 151:152] == 105 or data[i, 151:152] == 106 or data[i, 151:152] == 107:
#        newlabel[i, 0] = 2;
#    if data[i, 151:152] == 108 or data[i, 151:152] == 109:
#        newlabel[i, 0] = 3
#    if data[i, 151:152] == 110:
#        newlabel[i, 0] = 4;
#    if data[i, 151:152] == 111 or data[i, 151:152] == 112:
#        newlabel[i, 0] = 5;
#    if data[i, 151:152] == 113 or data[i, 151:152] == 114:
#        newlabel[i, 0] = 6;
#    if data[i, 151:152] == 115 or data[i, 151:152] == 116:
#        newlabel[i, 0] = 7;
print(np.unique(y))

use_ratio=0.1
x_use,x_unuse,y_use,y_unuse=train_test_split(x,y,test_size=1-use_ratio)

x_train,x_test,y_train,y_test=train_test_split(x_use,y_use,test_size=0.33)
scaler=preprocessing.StandardScaler().fit(x_use)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape)
print(y_train.shape)


#clf=SGDClassifier()
clf=svm.SVC(C=75,decision_function_shape='ovr')
#clf=DecisionTreeClassifier(max_depth=15)
#clf=RandomForestClassifier(n_estimators=100,max_depth=10)
#clf = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=2),
#    n_estimators=600,
#    learning_rate=1)
clf.fit(x_train,y_train)
train_acc=clf.score(x_train,y_train)
test_acc=clf.score(x_test,y_test)

print('train acc',train_acc)
print('test acc',test_acc)