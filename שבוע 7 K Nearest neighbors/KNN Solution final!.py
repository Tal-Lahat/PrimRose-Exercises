# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 07:40:48 2021

@author: eranb
"""
import numpy as np
from sklearn.model_selection import train_test_split

#1
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(150,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#2 & 3
    
def find_nearest(K,X,y,instance):
    X_res = np.apply_along_axis(lambda row: (((instance-row)**2).sum())**(1/2),1,X)
    X_sorted_by_index = np.argsort(X_res)
    top_Y_k = y[X_sorted_by_index][:K]
    return top_Y_k



def prediction(top_Y_k):
    counts = np.bincount(top_Y_k[:,0])
    return np.argmax(counts)

def predictator(X_test,X_train,y_train,K):
    predict_y = [prediction(find_nearest(K,X_train,y_train,instance)) for instance in X_test]
    res = np.array(predict_y)
    return res.reshape(len(res),1)


def accuracy_calc(y,predict_y):
    res = (y==predict_y)
    how_many_trues = np.count_nonzero(res)
    how_many_rows = len(res)
    return (how_many_trues*1./how_many_rows)*100

def main(X_test,X_train,y_test,y_train,K):
    print(accuracy_calc(y_test,predictator(X_test,X_train,y_train,K)))

main(X_test,X_train,y_test,y_train,20)
    