# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:52:04 2021

@author: shlom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



older_sibiling = np.asarray([31,22,40,26 ])
younger_sibiling = np.asarray([22,21,37,25])
times_talked = np.asarray([2,3,8,12])

data = pd.DataFrame({'older_sibiling':older_sibiling,'younger_sibiling':younger_sibiling,'times_talked':times_talked})

y=times_talked
x=np.concatenate([older_sibiling.reshape(len(older_sibiling),1),younger_sibiling.reshape(len(older_sibiling),1)],1)
xx=np.concatenate([np.ones(len(older_sibiling)).reshape(len(older_sibiling),1),older_sibiling.reshape(len(older_sibiling),1),younger_sibiling.reshape(len(older_sibiling),1)],1)

W1=np.linalg.pinv(x.T@x)@x.T@y
W2=np.linalg.pinv(xx.T@xx)@xx.T@y

L1=((x@W1.reshape(2,1)-y.reshape(4,1))**2).sum()
L2=((xx@W2.reshape(3,1)-y.reshape(4,1))**2).sum()


fig = plt.figure()
ax=Axes3D(fig) #axes 3d
# %matplotlib qt (to show figure)
# ax = fig.add_subplot(111, projection='3d')

xs=older_sibiling
ys=younger_sibiling
zs=y
ax.scatter(xs, ys, zs, marker='o',s=50)


# Make data.
X = np.arange(20, 40, 0.25)
Y = np.arange(20, 40, 0.25)
X, Y = np.meshgrid(X, Y)
f1 = W1[0]*X +W1[1]*Y
Z = f1
surf=ax.plot_surface(X,Y,Z) #axes 3d


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

diff= (older_sibiling-younger_sibiling)**2
xxx=np.hstack([xx,diff.reshape(len(diff),1)])
xxxx=xxx
xxx=xxx[:,1:4]
W3=np.linalg.pinv(xxx.T@xxx)@xxx.T@y

L3=((xxx@W3.reshape(3,1)-y.reshape(4,1))**2).sum()

W4=np.linalg.pinv(xxxx.T@xxxx)@xxxx.T@y
print(W3)
L4=((xxxx@W4.reshape(4,1)-y.reshape(4,1))**2).sum()
print(L1,L2,L3,L4)
#%%
# gradient descent

# h=w0+w1x1+w2x2
x0=np.ones((3,1))
x1=np.array([0.,1.,2.]).reshape(3,1)
x2=x1**2
y=np.array([1.,3.,7.]).reshape(3,1)
w=np.array([2.,2.,0.]).reshape(3,1)
wf=[[],[],[]]
loss=[[],[],[]]
m=1/len(x1)
X=np.hstack([x0,x1,x2])
a = [0.01,0.1,1]



for k in range(3):    
    wn=w.copy()
    for i in range(200):
        
        wn=wn-a[k]*m*(X.T@(X@wn -y))
       
    wf[k]=wn
    loss[k]=0.5*m*((X@wn-y)**2).sum()
print(wf[0],'\n',wf[1],'\n',wf[2],'\n')
print(f'this is a1 {loss[0]} a2 {loss[1]} a3 {loss[2]}')

# B -learning rate is too high
    






