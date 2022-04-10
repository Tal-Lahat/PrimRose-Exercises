# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 07:50:56 2021

@author: eranb
"""

#Solution for question 1

#a
import numpy as np
 
mu,sigma = 0 ,10
vec_X = np.random.normal(mu, sigma, 10)
random_ints = vec_X.astype(int)
#print(random_ints)

#b
vec_X2 = np.random.normal(mu,sigma,10)
#print(vec_X2)


#c
vec_X3 = (np.random.normal(mu,sigma,5)).astype(int)*3
#print(vec_X3)

#d bonus - a wierd solution that I created, used this link ->
# https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy/44308018

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
#get a random index via normal distribution between 1-10.    
rand_index = int(get_truncated_normal(mean=5, sd=2, low=1, upp=10).rvs())   
#print(rand_index)
rand_fibonacci = [1,1,2,3,5,8,13,21,34,55][rand_index]
#print(rand_fibonacci) 

#-----------Solution for question 2---------
#a
#first data set
first_array = vec_X*2
mu=0
sigma=2
noise1 =np.random.normal(mu, sigma, 10)
#print(first_array)
#b
first_array+=noise1
#print(first_array)


#c
#second data set

second_array = vec_X*2 + 4
mu=0
sigma=2
noise2 = np.random.normal(mu, sigma, 10)
second_array += noise2

#d
mu,sigma = 0 ,10
vec_X3 = np.random.normal(mu, sigma, 20)
third_array= vec_X3**2
#print(third_array)
mu=0
sigma=2
noise3 =np.random.normal(mu, sigma, 20)
third_array+=noise3

#Solution for question 3
#a
X1 = np.random.rand(4, 4)*10
X2 = np.random.rand(4,4)*10

#b
res = X1*X2

#c
inv_=np.linalg.inv(res)
tran_=res.T

#Solution for question 4

# Add constant feature to enable bias
XX = np.array([np.ones(vec_X.shape[0]), vec_X]).T

# Apply normal equations
Y1=first_array
W1 = np.linalg.inv(XX.T @ XX ) @ XX.T @ Y1

Y2=second_array
XX2=np.column_stack((np.ones(vec_X2.shape[0]),vec_X2))

##------##


XX3 = np.column_stack((vec_X3,vec_X3**2 ))
XX3 = np.column_stack((np.ones(XX3.shape[0]),XX3))
Y3=third_array
W3 = np.linalg.inv(XX3.T @ XX3 ) @ XX3.T @ Y3

import matplotlib.pyplot as plt

plt.scatter(vec_X,first_array)
plt.scatter(vec_X,XX@W1)
plt.show()


plt.scatter(vec_X3,third_array)
plt.scatter(vec_X3,XX3@W3)
plt.show()

#solution for question 8

X=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]

Y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]

X_train = np.array(X)
Y_log_train = np.log(np.array(Y))

XX_train = np.column_stack((X_train,X_train**2 ))
XX_train = np.column_stack((np.ones(XX_train.shape[0]),XX_train))

W_log = np.linalg.inv(XX_train.T @ XX_train ) @ XX_train.T @ Y_log_train

print(W_log)


#SOLUTION-----> log a = 1.38629436 ===> a = 4, b = 2 ,c=1
