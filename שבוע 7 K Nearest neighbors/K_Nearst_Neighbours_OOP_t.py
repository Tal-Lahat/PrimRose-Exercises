# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 01:01:03 2021

@author: Tal Lahat
K Nearst Neighbours - Exercise
Use train_test_split() to get training and test sets
Control the size of the subsets with the parameters train_size and test_size -
defult is 0.75/0.25 split
Determine the randomness of your splits with the random_state parameter -
any positive int will make your tests reproducible, none will result in random result
Obtain stratified splits with the stratify parameter
sklearn.model_selection module offers several other tools like-
from sklearn.linear_model import LinearRegression(), GradientBoostingRegressor(), and RandomForestRegressor(). GradientBoostingRegressor() and RandomForestRegressor() 
"""
# Object oriented  KNN
def main():
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn import model_selection
    import sklearn as sk
    
    class KNN:
        def __init__(self,x,y,k=3):
            self.x=x
            self.y=y
            self.k=k
            # self.X_train=X_train
            # self.X_test=X_test
            # self.y_train=y_train
            # self.y_test=y_test
            # self.dist=[]
            self.XY_train=[]
            self.XY_test=[]
            self.prediction=[]
            self.predictions=[]
            
        def train_test_split(self):
            model= sk.model_selection.train_test_split
            X_train, X_test, y_train, y_test = model(self.x, self.y,random_state=6)
            y_train=np.array(y_train.reshape((len(y_train),1)))
            y_test=np.array(y_test.reshape((len(y_test),1)))
            self.XY_train=np.concatenate([X_train,y_train],1) #I ended up just joining X and Y for easier sorting later on
            self.XY_test=np.concatenate([X_test,y_test],1)
            return self.XY_train,self.XY_test
        def Distance_function_2(self,all_train_rows,single_test_row):
            dist = np.sqrt(np.sum((all_train_rows-single_test_row)**2,axis=1))
            return dist
        def Nearest_neighbours(self,XY_train,test_row,k): #fit
            dist=self.Distance_function_2(self.XY_train[:,:-1],test_row).reshape((len(self.XY_train),1))  #First I caculate the distance
            dist=np.concatenate([self.XY_train,dist],1) #Now I add a distance column to my training data
            self.test1=dist
            dist=dist[dist[:,5].argsort()] #Now sorting by distance column the whole matrix, use argsort()][::-1] if you want sort by descent
            self.test=dist
            self.prediction=np.bincount(dist[:self.k,4].astype(int)).argmax() #Here I am counting which of the first K instances is the most frequent
            # print(f"Most frequent value for the {k} nearest neighbours is: {prediction}") 
            return self.prediction
        def accuracy(self,XY_train,XY_test,k): #predict
            ind=0
            percentage=0
            for test_row in XY_test[:,:-1]:
                self.prediction=self.Nearest_neighbours(self.XY_train,test_row,self.k).reshape(1)
                self.predictions = np.append(self.predictions, self.prediction, axis=0)
                expected=int(XY_test[ind,4])
                ind+=1
                if expected==self.prediction:
                    percentage+=1
                # print(f"Most frequent value for test row {test_row} for {k} nearest neighbours prediction is: {self.prediction} Vs expected [{expected}]\n") 
            print(f'Total sucess rate of prediction is {percentage*100/len(self.predictions)}%')
            return self.predictions    
    
    x,y=sk.datasets.load_iris(return_X_y=True)
    k=6
    Iris=KNN(x,y,k)
    X,Y=Iris.train_test_split()
    Predictions=Iris.accuracy(X,Y,k)
if __name__ =='__main__': # When importing a file, adding these two lines will prevent the script from running with out choosing what to run
    main()
    print(f'__name__: {__name__}')    
    #%%
    import import_for_name
    import_for_name.main()
    #%%