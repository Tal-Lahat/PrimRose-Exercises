# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:51:13 2021

@author: Tal
H.W Gradient Descent and normal equations Exercise
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
"""Normal Equations """
# Q1,A: Y=W1X1+W2X2
ages= pd.DataFrame({'Older sibling': [31,22,40,26],
                    'Younger sibling': [22,21,37,25],
                    'Times talked': [2,3,8,12]
    })
agesnp=ages.to_numpy()
x=agesnp[:,0:2]
y=agesnp[:,2]
y=y.reshape(4,1)

W=np.linalg.pinv(x.T @ x ) @ x.T @ y
print(W)
#%%
# Q1,B:  Y=W1X1+W2X2+W0*1 (X3=1) This new feature will allow more freadom (like not intersecting with 0 axis)
x=agesnp[:,0:2]
y=agesnp[:,2]
x1=np.concatenate([np.ones((len(x),1)),x],1) #X0,X1,X2
W1=np.linalg.pinv(x1.T @ x1 ) @ x1.T @ y
print(W1)
#Now lets check out what is our hypothesis is simillar to Y
W1=W1.reshape(3,1)
h1=(W1*x1.T).T #can also do x1@W1 for future refrence 

h1=np.sum(h1, axis=1)
err1=((y-h1)**2).min()
print('Error is:',err1)
#%%
# Q1,C:  Y=W1X1+W2X2+W3*X3 (X3=[X1-X2]^2) This new feature will add a feature that is linearly independent  
x=agesnp[:,0:2]
y=agesnp[:,2]
xms=np.concatenate([x,((x[:,0]-x[:,1])**2).reshape(4,1)],1) #X1,X2,[X1-X2]^2
Wms=np.linalg.pinv(xms.T @ xms ) @ xms.T @ y
print(Wms)
#Now lets check out what is our hypothesis is simillar to Y
Wms=Wms.reshape(3,1)
hms=(Wms*xms.T).T
hms=np.sum(hms, axis=1)
err_ms=((y-hms)**2).min()
print('Error is:',err_ms)
#%%
# Q1,E?:  Y=W0X0+W1X1+W2X2+W3*X3 (X3=[X1-X2]^2) all privious features
xall=np.concatenate([x1,((x[:,0]-x[:,1])**2).reshape(4,1)],1) #X0,X1,X2,[X1-X2]^2
Wall=np.linalg.pinv(xall.T @ xall ) @ xall.T @ y
print(Wall)
#Now lets check out what is our hypothesis is simillar to Y
Wall=Wall.reshape(4,1)
hall=(Wall*xall.T).T
hall=np.sum(hall, axis=1)
err_all=((y-hall)**2).min()
print('Error is:',err_all)
# wow! the error is nearly zero!

#%% Loss function: (H(x)-Y)^2
LA=((x@W-y.reshape(4,1))**2).sum()
LB=((x1@W1-y)**2).sum()
LC=((xms@Wms-y)**2).sum()
LE=((xall@Wall-y)**2).sum()

print('Ls for question 1A=',LA,'\nLs for question 1B=',LB)
print('Ls for question 1C=',LC,'\nLs for question 1E=',LE)
test=[LA,LB,LC,LE]
print('The smallest one is equal to ',min(test))
#%%
"""Gradient Descent"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Q2,A: h=W0+W1X1+W2*(X1**2)  (X1**2) =X2
X0=np.array([[1.0],[1.0],[1.0]])
X1=np.array([[0.0],[1.0],[2.0]])
X2=X1**2.0
X=np.concatenate([X0,X1,X2],1)
Y=np.array([[1.0],[3.0],[7.0]])
LR=[0.01, 0.1, 1.0] #Learning rate, in formula its alpha
iterations=200
W=np.array([[2.0],[2.0],[0.0]])
M=(X.shape[0])
def Loss_function_MSE(X,W,Y):
    Ls=(1.0/(2.0*M))*(((X@W-Y)**2.0).sum())
    return Ls
print("First Loss function is equal to:",Loss_function_MSE(X,W,Y),'\n')
"""Gradient descent is basicly a derivative on the Loss function"""
def Gradient_Descent(X,W,Y):
    for i in range(iterations):
        W=W-(
            lr*(
                (1/M)   *    (X.T@(X@W-Y))
                )
            )
    
    print("Ls for LR=",lr,"& after 200 iterations is:",Loss_function_MSE(X,W,Y))
    print("Final W is equal to:\n"+str(W)+'\n\n')
    # return W
for lr in LR:
    Gradient_Descent(X,W,Y)
#Q2,B:
"""For LR=0.01 it was slower but successful, for LR=0.1 we recived the best loss result, while LR=1 overshoots"""
W=np.linalg.pinv(X.T @ X ) @ X.T @ Y
print("The theta result from Linear regression is:\n",W)