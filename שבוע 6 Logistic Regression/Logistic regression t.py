# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:36:21 2021

@author: Tal
H.W Logistic regression classifirer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

#Q1-1: a. If the last column value is ‘Iris-setosa” - 1 ,b. Else - 0

iris = datasets.load_iris()
x = iris.data[:, :2]  # we only take the first two features.
y = iris.target
y=np.where(y==0,1,0)
# plotting the data
fig=plt.figure(2, figsize=(8, 6))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
colormap=np.array(['r','b'])
plt.scatter(x[:, 0],x[:, 1],c=colormap[y])
plt.show()
y=y.astype(float).reshape(150,1)
#%%
#Q2-1,2,3:Creates Logit function, Create Loss function (likelihood), create gradient function
iterations=1000
LR=0.01
m=(x.shape[0])
eps=0.0001 #To prevent log 0

if np.shape(x)[1]==2: #Just so it wont add a column of ones again if I run this section again
    x=np.concatenate([np.ones(len(x)).reshape(len(x),1),x],1)
# Ws=np.random.normal(loc=1,scale=1.5,size=3).reshape(3,1) #do I need a 1 vector? This is the starting W
Ws=np.array([0.9,0.6,0.4]).reshape(3,1) #for testing
W=np.copy(Ws)

def logit (W,x): #sigmoid
    z=x@W
    return 1/(1+np.exp(-z))

def log_loss(W,x,y): #Log likelihood
    g=logit(W,x)
    return (-1/m)*(
                  y.T @ np.log(g+eps) + (1-y).T @np.log(1-g+eps)
                  )

def logistic_regression_gradient(W,x,y):
    loss_plot=np.zeros(iterations)
    for i in range(iterations):
        g=logit(W,x)
        W-=(LR/m)*x.T @ (g-y)
        # Now to be able to plot loss later
        loss_plot[i]=log_loss(W,x,y)
    return W,loss_plot
Wfinal,loss_plot=logistic_regression_gradient(W,x,y)
loss_s=log_loss(Ws,x,y)
loss_f=log_loss(W,x,y)
print('The Likelihood before Gradient and iteratioins on W is',loss_s,'The Likelihood after Gradient is',loss_f)


#Here I am testing and seing how the hypothesis is similar to y by turning hypothesis to zero or one.
xl=np.linspace(0,10,150)
yl=Wfinal[0]*x[:, 0]+Wfinal[1]*x[:, 1]+Wfinal[2]*x[:, 2]
def NormalizeData(data):
    return (data - np.min(yl)) / (np.max(yl) - np.min(yl))

scaled_yl = NormalizeData(yl)

print(scaled_yl)
asign = lambda t: 0 if t<0.5 else 1
scaled_yl_zero_one = list(map(asign, scaled_yl))
print(scaled_yl_zero_one)    
#%% Splitting
def split(p,x,y):
    # p=np.random.randint(low=1,high=100) #1-100, this is how we split the data to train and test.
    permutation=np.random.permutation(m)
    x=x[permutation]
    y=y[permutation]
    p=p/100
    X_train=x[:int(round(150*p, 0)),]
    X_test=x[int(round(150*p, 0)):,]
    Y_train=y[:int(round(150*p, 0)),]
    Y_test=y[int(round(150*p, 0)):,]
    return (X_train,X_test,Y_train,Y_test)
p=60 
X_train,X_test,Y_train,Y_test=split(p,x,y)
#%% Training
def train():
    Ws=np.random.normal(loc=1,scale=1.5,size=3).reshape(3,1) 
    # Ws=np.array([0.9,0.6,0.4]).reshape(3,1) #for testing
    W=np.copy(Ws)
    W_train,loss_plot=logistic_regression_gradient(W,X_train,Y_train)
    plt.plot(loss_plot)
    loss_train=log_loss(W_train,X_train,Y_train)
    return W_train,loss_train
W_train,loss_train=train()    
#%% Testing

def testing_train():
    g=logit(W_train,X_train)
    pred=np.zeros_like(g)
    pred[g>=0.5]=1
    accuracy=100*sum([x==y for x,y in zip(pred,Y_train)])/len(Y_train)
    # loss_test=log_loss(W_train,X_test,Y_test)
    print(f"precision:{accuracy}%, log loss train:{loss_train}")
    return loss_test
loss_test=testing_train() 
#%% Testing

def testing_test():
    g=logit(W_train,X_test)
    pred=np.zeros_like(g)
    pred[g>=0.5]=1
    accuracy=100*sum([x==y for x,y in zip(pred,Y_test)])/len(Y_test)
    loss_test=log_loss(W_train,X_test,Y_test)
    print(f"precision:{accuracy}%, log loss test:{loss_test}")
    return loss_test
loss_test=testing_test() 

#%% Testing original

# def testing():
#     g=logit(W_train,x)
#     pred=np.zeros_like(g)
#     pred[g>=0.5]=1
#     accuracy=100*sum([x==y for x,y in zip(pred,y)])/len(y)
#     loss_test=log_loss(W_train,X_test,Y_test)
#     print(f"precision:{accuracy}%, log loss test:{loss_test}")
#     return loss_test
# loss_test=testing() 
#%%
"""Example of using Axes 3d"""
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets
# from sklearn.decomposition import PCA

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target

# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# plt.figure(2, figsize=(8, 6))
# plt.clf()

# # Plot the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
#             edgecolor='k')
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())

# # To getter a better understanding of interaction of the dimensions
# # plot the first three PCA dimensions
# fig = plt.figure(1, figsize=(8, 6))
# # ax = Axes3D(fig, elev=-150, azim=110)
# ax = fig.add_subplot(111, projection='3d')
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
#             cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])

# plt.show()