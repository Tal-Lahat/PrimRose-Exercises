
#%%
import import_for_name
import_for_name.main()
#%%

#%%The old way part 1
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
import sklearn as sk
x,y=sk.datasets.load_iris(return_X_y=True)
model= sk.model_selection.train_test_split
X_train, X_test, y_train, y_test = model(x, y,random_state=6)
y_train=np.array(y_train.reshape((len(y_train),1)))
y_test=np.array(y_test.reshape((len(y_test),1)))
XY_train=np.concatenate([X_train,y_train],1) #I ended up just joining X and Y for easier sorting later on
XY_test=np.concatenate([X_test,y_test],1)

#%%The old way part 2
#1. Handle the data option B, Downloading the data and use np.genfromtext, not using this method this time:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# iris=np.genfromtxt(r'C:\Users\leora\Desktop\טל\קורס MACHINE LEARNING\פרימרוז\שבוע 7\iris.txt', dtype='unicode',delimiter=',')

#2. Distance function second way:
def Distance_function_2(all_train_rows,single_test_row):
    dist = np.sqrt(np.sum((all_train_rows-single_test_row)**2,axis=1))
    return dist
# dist2=Distance_function_2(x,x[1,:]) #testing
#%%The old way part 3
#3.A&B Nearest neighbours:
def Nearest_neighbours(XY_train,test_row,K):
    dist=Distance_function_2(XY_train[:,:-1],test_row).reshape((len(XY_train),1))  #First I caculate the distance
    dist=np.concatenate([XY_train,dist],1) #Now I add a distance column to my training data
    dist=dist[dist[:,5].argsort()] #Now sorting by distance column the whole matrix, use argsort()][::-1] if you want sort by descent
    prediction=np.bincount(dist[:K,4].astype(int)).argmax() #Here I am counting which of the first K instances is the most frequent
    # print(f"Most frequent value for the {K} nearest neighbours is: {prediction}") #testing
    return prediction

# prediction=Nearest_neighbours(XY_train,XY_test[0,:-1],10) #testing
#%%The old way part 4
#4. Calculate the accuracy on the test data (and Nearest_neighbours on all test values ):
def accuracy(XY_train,XY_test,K):
    predictions = []
    ind=0
    percentage=0
    for test_row in XY_test[:,:-1]:
        prediction=Nearest_neighbours(XY_train,test_row,K).reshape(1)
        predictions = np.append(predictions, prediction, axis=0)
        expected=int(XY_test[ind,4])
        ind+=1
        if expected==prediction:
            percentage+=1
        # else:
        #     print('flag')
        print(f"Most frequent value for test row {test_row} for {K} nearest neighbours prediction is: {prediction} Vs expected [{expected}]\n") 
    print(f'Total sucess rate of prediction is {percentage*100/len(predictions)}%')
    return predictions
    
K=7
predictions=accuracy(XY_train,XY_test,K)


# ind=0
# percentage=0
# for test_row in XY_test[:,:-1]:
#     self.prediction=self.Nearest_neighbours(self.XY_train,test_row,self.k).reshape(1)
#     self.predictions = np.append(self.predictions, self.prediction, axis=0)
#     expected=int(XY_test[ind,4])