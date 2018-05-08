# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas

data_path='/media/sdb/xiaobingdu/data_bmovie2.mat'
#load dataset
datamat = sio.loadmat(data_path)
data = datamat['xlsdata']
print(data.shape)

y_col=151
x_raw=data[:,:150]
y_raw=data[:,y_col]

labels=np.unique(y_raw)
n=len(labels)
train_results=np.zeros((n,n))
test_results=np.zeros((n,n))
for i,label_a in enumerate(labels):
    for j,label_b in enumerate(labels):
        if label_a == label_b:
            continue
        
        y=np.zeros_like(y_raw)
        if y_col == 151:
            y[y_raw==label_a]=1
            y[y_raw==label_b]=2
            x=x_raw[y>0,:]
            y=y[y>0]
            
        
        print('label is',label_a,label_b)
        print('unique is',np.unique(y))
        
        use_ratio=0.95
        x_use,x_unuse,y_use,y_unuse=train_test_split(x,y,test_size=1-use_ratio)
        
        x_train,x_test,y_train,y_test=train_test_split(x_use,y_use,test_size=0.33)
        scaler=preprocessing.StandardScaler().fit(x_use)
        
        x_train=scaler.transform(x_train)
        x_test=scaler.transform(x_test)
        
        print('x_train shape is',x_train.shape)
        print('y_train shape is',y_train.shape)
        
        clf=svm.SVC(C=1.0)
        #clf=DecisionTreeClassifier(max_depth=15)
        clf.fit(x_train,y_train)
        train_acc=clf.score(x_train,y_train)
        test_acc=clf.score(x_test,y_test)
    
        print('train acc',train_acc)
        print('test acc',test_acc)
        
        train_results[i,j]=train_acc
        test_results[i,j]=test_acc

print('train results',np.round(train_results,decimals=2))
print('test result',np.round(test_results,decimals=2))

a=pandas.DataFrame()
c=a.from_dict(np.round(train_results,2))
c.columns=np.unique(y_raw)
c.index=np.unique(y_raw)
d=a.from_dict(np.round(test_results,2))
d.columns=np.unique(y_raw)
d.index=np.unicode(y_raw)

print('train results'+'*'*30)
print(c)
print(c.mean(axis=0))
print('test result'+'*'*30)
print(d)
print(d.mean(axis=0))