# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:54:55 2021

@author: shlom
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

data=pd.read_csv(url,names=['sepal_length','sepal_width','petal_length','petal_width','class'])

#%%


def name_change(in_str,keyword):
    if in_str ==keyword:
        return 1
    else:
        return 0

def sigmoid(x):
    # import numpy as np
    return 1/(1+np.exp(-x))

# def cost_function(h,y,m):
#     return (1/m) * (-y.T @ np.log(h) - (1-y).T @ np.log(1-h))
# def gradient_function(h,y,m,x):
#     (1/m)*((h-y)@x)

def rand_picker(x,percentage):
    
    percentage=percentage/100
    end =percentage*len(x)
    end =int(end)
    numbers=list(range(len(x)))
    new_arr=np.zeros(len(x))
    
    for i in range(len(x)):
        new_arr[i]=numbers.pop(np.random.choice(len(numbers)))    
    
    x=x[new_arr.astype('int')]
    train=x[0:end]
    test=x[end:]
    return([train,test])
    
# creating a column of 0 - 1 according to species

data['numeric_species']=data['class'].apply(name_change,keyword='Iris-setosa')

# plotting the data
fig1=plt.figure()
x1=data.iloc[:,0]
x2=data.iloc[:,1]
categories=data.loc[:,'numeric_species']
colormap=np.array(['r','b'])
plt.scatter(x1,x2,c=colormap[categories])
plt.show()

# applying random test/train data
test,train=rand_picker(data,75)





plt.close('all')