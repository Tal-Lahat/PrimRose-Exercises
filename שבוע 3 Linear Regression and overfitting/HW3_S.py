import numpy as np
import matplotlib.pyplot as plt

def fibonacci(size):
    size =size-2
    result = [0,1]
    for i in range(size):
        result.append(result[1+i]+result[i])
    return result

#########################################################
#1-1 random 10 integers 
num1 = np.random.randint(1,21,10)
print(num1)

# 1-2 random 10 float
num2 = np.random.random(10)
print(num2)

# 1-3 three random multiples of 3
num3 = 3*np.random.randint(1,21,5)
print(num3)

#1-4 a random number from the first 10 in fibonacci series
# fib=fibonacci(10)
num4=np.random.choice(np.array(fibonacci(10)),10)
print(num4)

#########################################################
# 2-1 10 random points on y=ax

a1=1
x1 = 10*(np.random.random(10)-0.5) #10 random points within 5 units from the origin
n1=np.random.normal(0,1,10) # noise max 1 unit
first_array =a1*x1+n1
plt.scatter(x1,first_array)
plt.show()

#%%
# 2-2 ax+b+noise
a2=1
b2=5*np.random.random(1)*np.ones(10)
x2 = 10*(np.random.random(10)-0.5) #10 random points within 5 units from the origin
n2=np.random.normal(0,1,10) # noise max 1 unit
second_array =a2*x2+b2+n2
plt.scatter(x2,second_array)
plt.show()

#%%
# 2-3 y=x^2
a3=-4
b3=3
c3=-5


x3 = 10*(np.random.random(10)-0.5) #10 random points within 5 units from the origin
n3=np.random.normal(0,1,10) # noise max 1 unit
third_array = a3*x3**2 + b3*x3 + c3*np.ones(10) +n3
plt.scatter(x3,third_array)
plt.show()

#%%
#########################################################
# 3-1 10 random points on y=ax
A=np.random.randint(-10,11,[4,4])
B=np.random.randint(3,20,[4,4])
C=A @ B
print(C,'\n',C.T,'\n',np.linalg.pinv(C))
#########################################################
#%%
# 4-1


a1=5
x1=10*(np.random.random(10)-0.5)
n1=8*(np.random.random(10)-0.5)
y1=a1*x1+n1
#y1_ideal=a1*x1
#y1_ideal=y1_ideal.reshape([10,1])
x1 = np.reshape(x1,[10,1])
y1 = np.reshape(y1,[10,1])
xf1=np.concatenate([np.ones(10).reshape([10,1]),x1],1)
W=(np.linalg.pinv(xf1.T @ xf1)) @ (xf1.T @ y1)
plt.scatter(x1,y1)
x1_ls=np.linspace(-5,5,50)
plt.plot(x1_ls,W[1]*(x1_ls)+W[0])
plt.show
print(W)

#%%
# 4-2

# a2=1
# b2=5*np.random.random(1)*np.ones(10)
# x2 = 10*(np.random.random(10)-0.5) #10 random points within 5 units from the origin
# n2=np.random.normal(0,1,10) # noise max 1 unit
# second_array =a2*x2+b2+n2


x2 = np.reshape(x2,[10,1])
b2 = np.reshape(b2,[10,1])
n2 = np.reshape(n2,[10,1])
y2 = np.reshape(second_array,[10,1])
xf2= np.concatenate([np.ones(10).reshape([10,1]),x2],1)
W2=(np.linalg.pinv(xf2.T @ xf2)) @ (xf2.T @ y2)
plt.scatter(x2,y2)
x2_ls=np.linspace(-5,5,50)
plt.plot(x2_ls,W2[1]*(x2_ls)+W2[0])
plt.show
#%%
a3=-4
b3=3
c3=-5


x3 = 10*(np.random.random(10)-0.5) #10 random points within 5 units from the origin
n3=np.random.normal(0,1,10) # noise max 1 unit
third_array = a3*x3**2 + b3*x3 + c3*np.ones(10) +n3
plt.scatter(x3,third_array)
plt.show()

x3 = np.reshape(x3,[10,1])
n3 = np.reshape(n3,[10,1])
y3 = np.reshape(third_array,[10,1])

xf3= np.concatenate([np.ones(10).reshape([10,1]),x3,x3**2],1)
W3 = (np.linalg.pinv(xf3.T @ xf3)) @ (xf3.T @ y3)
plt.scatter(x3,y3)
x3_ls=np.linspace(-5,5,50)
plt.plot(x3_ls,W3[2]*x3_ls**2+W3[1]*(x3_ls)+W3[0])
plt.show
#%%

x=np.array([0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823])
x=np.reshape(x,[len(x),1])
xx= np.concatenate([np.ones(len(x)).reshape([len(x),1]),x,x**2],1)
y=np.array([4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882])
y=np.reshape(y,[len(y),1])
y=np.log(y)
print(xx,y)
WW = (np.linalg.pinv(xx.T @ xx)) @ (xx.T @ y)
x_ls=np.linspace(np.floor(np.min(x))-1,np.ceil(np.max(x))+1,50)
a=np.e**WW[0]
b=WW[2]
c=WW[1]

print(a,b,c)
# a=4
# b=2
# c=1
