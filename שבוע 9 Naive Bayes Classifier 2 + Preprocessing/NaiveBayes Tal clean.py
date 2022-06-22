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
from sklearn.naive_bayes import GaussianNB #To compare to Sklearn Naive Bayes results
from scipy import stats  #Will use to remove outliers
from sklearn import preprocessing #Will be used to map labels
#1. Handle the Data
# choose dataframe:
# data=pd.read_csv(r"D:\קורס MACHINE LEARNING\פרימרוז\שבוע 9 Naive Bayes Classifier 2 + Preprocessing\iris.csv", sep=',')
# data=pd.read_csv(r"D:\קורס MACHINE LEARNING\פרימרוז\שבוע 9 Naive Bayes Classifier 2 + Preprocessing\testdata.csv", sep=',')
data=pd.read_csv(r"D:\קורס MACHINE LEARNING\פרימרוז\שבוע 9 Naive Bayes Classifier 2 + Preprocessing\diabetes.csv", sep=',')

#Preprocessing:
le = preprocessing.LabelEncoder() #Notice! this will remove string and turn it into int for encoding e.g Setosa:1,Versicolor:2,Virginica:3
data.iloc[:,-1]=le.fit_transform(data.iloc[:,-1]) #Another option will be to use np.unique(data.iloc[:,-1]) or set(data.iloc[:,-1])

for i in range(1,data.shape[1]-1):  #Used to change zeroes
    median_i=data.iloc[:,i].median()
    data.iloc[:,i]=data.iloc[:,i].apply(lambda x: median_i if x==0 else x)
    
data=data[(np.abs(stats.zscore(data)) < 5).all(axis=1)]  #Used to remove outliers, I put 5 but 3 is a more conservative option

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)] #Used to remove aditional outliers

train,test=model(data,test_size=0.3,random_state=8) #if we want to split to train test with out spliting to data and labels ,random_state=8
#2. Summarize the Data (train)
classes=train.groupby(data.columns[-1])
classes=list(classes.groups)
classes=np.array(classes)
data_train_mean=train.groupby(data.columns[-1]).mean() #change data.groupby to train.groupby or test.groupby to change on what to train
data_train_std=train.groupby(data.columns[-1]).std() 
# 3. Write a prediction function :
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
 	exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
 	return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

features=train.shape[1]-1 #Number of features, change data if needed

# 4. Make Predictions :
listoflabels=test[train.columns[-1]].values
sucess=0
for row in range(np.shape(test)[0]):
    x=test.iloc[row,:-1].values.astype(float).reshape(1,features) #check all rows
    answers=[]
    for Num_classes in range(np.size(classes)):
        probability_each_feature=calculate_probability(x, data_train_mean.iloc[Num_classes,:].values.reshape(1,features), data_train_std.iloc[Num_classes,:].values.reshape(1,features))
        Pclasses=train.groupby(train.columns[-1]).size()[Num_classes]/len(train) #caculating the probability of a class from all classess
        # print(Pclasses*np.prod(probability_each_feature)) #for debuging
        answers.append(Pclasses*np.prod(probability_each_feature))
        if np.size(classes)==Num_classes+1:
            answer=classes[np.argmax(answers)]
            if answer==listoflabels[row]:
                sucess+=1
print('Sucess rate for us is:',(sucess/len(test))*100,'%')
clf=GaussianNB()
clf.fit(train.iloc[:,:-1], train.iloc[:,-1])
accuracy_sklearn=clf.score(test.iloc[:,:-1], test.iloc[:,-1])
print('Accuracy of Sklearn is: ',accuracy_sklearn*100,'%')