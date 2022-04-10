## -*- coding: utf-8 -*-
"""
linear regression H.W
Created on Fri Feb  5 21:41:03 2021

@author: Tal
"""
import numpy as np

"""
#Question 1
"""
#Exercise a: Create a vector from 10 random integers 
vector=np.random.randint(low=0,high=100,size=10)
print(vector,'\n')
#Exercise b: Create a vector named x from 10 random floats 
X=np.random.normal(loc=4,scale=5,size=10)
print(X,'\n')
#Exercise c: Create a vector from 5 random multiple of 3 integers 
TimesThree=np.random.randint(low=0,high=100,size=5)*3
print(TimesThree,'\n')
#Exercise d: Choose a random number from the first 10 Fibonacci numbers (?)*
Fib=np.random.choice(np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),1)
print(Fib,'\n')
#*Not sure what the original exercise meant... 


#%%

"""
#Question 2

"""
import numpy as np
from matplotlib import pyplot as plt
#Exercise a: Create a vector named first_array from x that represents 10 randoms dots on an inclined stright line through the origion
X=np.random.normal(loc=4,scale=5,size=10)
#Exercise b: add "Noise":
first_array= 2*X + np.random.normal(loc=0,scale=1,size=10) ##Y=AX+noise
print(first_array,'\n')
# plt.plot(X, y, 'r')
plt.scatter(X, first_array)
plt.show()
#%%
#Exercise c: Create a second_array similaraly to b (different dots) but not through the origion:
XX=np.random.normal(loc=4,scale=7,size=10)
second_array= 3*XX+np.random.randint(10) + np.random.normal(loc=0,scale=1,size=10) + np.random.randint(low=3,high=12,size=1)*np.ones(10) ##Y=AX+B+noise 
print(second_array,'\n')
plt.scatter(XX, second_array)
plt.show()
 
#%%
#Exercise c: Create a Third_array,20 dots, a parabola (with noise too):
XXX=np.random.normal(loc=0,scale=7,size=20)
third_array= (2*(XXX+np.random.randint(10)))**2 -3*XXX+ np.random.normal(loc=0,scale=1.3,size=20) ##Y=AX^2+BX+noise
print(third_array,'\n')
plt.scatter(XXX, third_array)
plt.show()
#%%
"""
#Question 3

"""
import numpy as np
#Exercise a: Create 2 4x4 matrices
# matrix_a=np.array([[np.random.randint(low=-10,high=10,size=4)],[np.random.randint(low=-10,high=10,size=4)],[np.random.randint(low=-10,high=10,size=4)],[np.random.randint(low=-10,high=10,size=4)]])
matrix_a=np.random.rand(4,4) #and the easy way for float
matrix_b=np.random.randint(low=-10,high=10,size=(4,4)) #and the easy way for int
#Exercise a: multiple 2 4x4 matrices
matmul=matrix_a @ matrix_b
print('Matrix multiplication:\n',matmul)
#Exercise a: Invese and Transpose of exercise b result.
matinv=np.linalg.pinv(matmul)
print('Matrix Inverse:\n',matinv)
mattrans=matmul.T
print('Matrix Transpose:\n',mattrans)

#%%
"""
#Question 4+7

"""
#1 First_array
X1=np.array([np.ones(X.shape[0]), X]).T
# first_array = np.reshape(first_array,[10,1]) #This line is not necessary, but can be activated making sure the shape is as required and using reshape like the question proposed I solved this.
W=np.linalg.pinv(X1.T @ X1 ) @ X1.T @ first_array
plt.scatter(X,first_array)
X1_ls=np.linspace(-5,30,500)
plt.plot(X1_ls,W[1]*(X1_ls)+W[0],'r')
plt.show
print(W)

#%%
"""
#Question 5+7

"""
#2 Second_array
# XX1=np.array([np.ones(XX.shape[0]), XX]).T       #This is how I did it previously, and it works, but will do it with column stack as required
XX1=np.column_stack((np.ones(XX.shape[0]), XX)) 
# second_array = np.reshape(second_array,[10,1])   #This line is not necessary, but can be activated making sure the shape is as required and using reshape like the question proposed I solved this
W2=np.linalg.pinv(XX1.T @ XX1 ) @ XX1.T @ second_array
plt.scatter(XX,second_array)
XX1_ls=np.linspace(-5,30,500)
plt.plot(XX1_ls,W2[1]*(XX1_ls)+W2[0],'r')
plt.show
print(W2)
#%%
"""
#Question 6 +7
"""
XXX1=np.array([np.ones(XXX.shape[0]), XXX, XXX**2]).T
# third_array = np.reshape(third_array,[20,1])      #This line is not necessary, but...
W3=np.linalg.pinv(XXX1.T @ XXX1 ) @ XXX1.T @ third_array
plt.scatter(XXX,third_array)
XXX1_ls=np.linspace(-10,10,20)
plt.plot(XXX1_ls,W3[2]*XXX1_ls**2+W3[1]*(XXX1_ls)+W3[0],'r')
plt.scatter(XXX, third_array)
plt.show()
print(W3)
#%%
"""
#Question 8
y=a*EXP^(bx^2)+cx

x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.690
21229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9
591165,0.9496439 ,0.82249202,0.99367066,0.50628823]
y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.6
8259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487
443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]

Find a,b,c
"""
X=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]

Y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]

X_train = np.array(X)
Y_log_train = np.log(np.array(Y))

XX_train = np.column_stack((X_train,X_train**2 ))
XX_train = np.column_stack((np.ones(XX_train.shape[0]),XX_train))

W_log = np.linalg.inv(XX_train.T @ XX_train ) @ XX_train.T @ Y_log_train

print(W_log)
#%%

"""
How it was solved in the lecture
"""

# create features for order 2 and 3 terms
X=np.random.normal(loc=0,scale=7,size=20)
Y = (3*(X-0.5))**3 + 5*(np.random.randn(20))
XX = np.array([np.ones(X.shape[0]), X, X**2, X**3]).T

# Apply normal equations
W = np.linalg.inv(XX.T @ XX ) @ XX.T @ Y

print(W)
# just a grid of x coordinates
x = np.linspace(-10, 10, 1000)  

# create features for order 2 and 3 terms
xx = np.array([np.ones(x.shape[0]), x, x**2, x**3]).T

# apply hypothesis to feature grid
y = xx @ W.T  

# Plot training points and hypothesis
plt.scatter(X, Y)
plt.plot(x, y, 'r')
plt.show()
