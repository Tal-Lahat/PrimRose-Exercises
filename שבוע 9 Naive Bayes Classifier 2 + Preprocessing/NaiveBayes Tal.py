# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:54:45 2021

@author: Tal
NaiveBayes
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import sklearn as sk
model= sk.model_selection.train_test_split

#1. Handle the Data
# load another dataframe:
data=pd.read_csv(r"C:\Users\leora\Desktop\טל\קורס MACHINE LEARNING\פרימרוז\שבוע 9\iris.csv", sep=',')
# data=pd.read_csv(r"C:\Users\leora\Desktop\טל\קורס MACHINE LEARNING\פרימרוז\שבוע 9\testdata.csv", sep=',')
# data=pd.read_csv(r"C:\Users\leora\Desktop\טל\קורס MACHINE LEARNING\פרימרוז\שבוע 9\diabetes.csv", sep=',')

# X = data.iloc[:,:-1].values #.values transform the dataframe data to numpy data with the lowest common denominator format
# y = data.iloc[:,-1].values
# X_train, X_test, y_train, y_test = model(X, y,random_state=6)  #choosing a random_state makes the shuffle not random, good for testing purposes

train,test=model(data,test_size=0.3,random_state=6) #if we want to split to train test with out spliting to data and labels
# X_train = train.iloc[:,:-1].values 
# y_train = train.iloc[:,-1].values
# X_test = test.iloc[:,:-1].values 
# y_test = test.iloc[:,-1].values

#2. Summarize the Data (train)
classes=data.groupby(data.columns[-1])
classes=list(classes.groups)
classes=np.array(classes)
data_train_mean=data.groupby(data.columns[-1]).mean() #change data.groupby to train.groupby or test.groupby to change on what to train
data_train_std=data.groupby(data.columns[-1]).std() 
# classes=pd.unique(data.iloc[:,-1]) #change data to train to test or whatever, this is the different classes 
# count_classes=data.iloc[:,-1].nunique()
# 3. Write a prediction function :
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

features=data.shape[1]-1 #Number of features, change data if needed

# 4. Make Predictions :
sucess=0
for row in range(np.shape(data)[0]):
    x=data.iloc[row,:-1].values.astype(float).reshape(1,features) #check all rows
    # answer=np.empty((np.size(classes),1))
    answers=[]
    for Num_classes in range(np.size(classes)):
        probability_each_feature=calculate_probability(x, data_train_mean.iloc[Num_classes,:].values.reshape(1,features), data_train_std.iloc[Num_classes,:].values.reshape(1,features))
        Pclasses=data.groupby(data.columns[-1]).size()[Num_classes]/len(data) #caculating the probability of a class from all classess
        print(Pclasses*np.prod(probability_each_feature))
        answers.append(Pclasses*np.prod(probability_each_feature))
        if np.size(classes)==Num_classes+1:
            answer=classes[np.argmax(answers)]
            if answer==data[data.columns[-1]][row]:
                sucess+=1
print('Sucess rate is:',(sucess/len(data))*100,'%')
           # print(classes[np.argmax(answers)])
           
    # np.argmax(answer) 
    # max(answer)       
        # print(answer)

# blabla=data[data.columns[-1]==0]
# data[data.columns[-1]]==0
# (data[data.columns[-1]]==0).sum()
# (data[data.columns[-1]]==1).sum()
# .quantity.sum()