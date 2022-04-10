


import numpy as np
from matplotlib import pyplot as plt

NUM_POINTS = 100

random_ints = np.random.randint(0,10,10)
ranom_floats = 2*np.random.rand(NUM_POINTS)-1
random_3_multipliers = np.random.randint(0,10,10) * 3

print('Random integers: ', random_ints)
print('Random floats: ', ranom_floats)
print('Random 3 multipliers: ', random_3_multipliers)

def fibonachi(n):
    if n==1 or n==0:
        return 1
    return fibonachi(n-1)+fibonachi(n-2)

fibo_index=np.random.randint(0,10)
random_fibo = fibonachi(fibo_index)
print('Random fibonachi at location {} is: {}'.format(fibo_index, random_fibo))

#creating random data
x = ranom_floats

m_1 = 2
first_array = m_1*x
gaussian_noise = np.random.normal(0,0.2,NUM_POINTS)
first_array += gaussian_noise

m_2 = 3
n_2 = -1
second_array = m_2*x + n_2 * 1
second_array += gaussian_noise

l_3 = 2
m_3 = 1
n_3 = 0.5
third_array = l_3*(x**2) + m_3*x + n_3
third_array += gaussian_noise

#exercise calculating matrix operations
mat1 = np.random.rand(4,4)
mat2 = np.random.rand(4,4)

#two solutions:
mult= np.dot(mat1,mat2)
mult = mat1@mat2

#two ways to od transpose
mult.transpose()
mult.T

print('Matrix inverse ', np.linalg.inv(mult))


def normal_equations(x,y):
    assert x.ndim==2, 'input shape must be two dimensional'
    return np.linalg.inv(x.T@x)@x.T@y
 
#first array
x=x.reshape(len(x),1) 
slope = normal_equations(x,first_array)

#second array
ones_vec = np.ones((NUM_POINTS,1))
x2=np.column_stack((ones_vec,x))
reg2_res=normal_equations(x2,second_array)
a = reg2_res[1]
b = reg2_res[0]

plt.figure()
plt.scatter(x,second_array,s=2,c='r')
x_of_line = np.linspace(-1,1,11)
y_of_line = a*x_of_line+b
plt.plot(x_of_line,y_of_line )

#third array
x3=np.column_stack((ones_vec,x, x**2))
reg3_res=normal_equations(x3,third_array)

plt.figure()
plt.scatter(x,third_array,s=2,c='r')
y_of_parabola = (x_of_line**2)*reg3_res[2]+x_of_line*reg3_res[1]+reg3_res[0]
plt.plot(x_of_line,y_of_parabola )



x=[0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,0.54419358,0.89099501,0.9591165,0.9496439 ,0.82249202,0.99367066,0.50628823]
x=(np.array(x)).reshape(len(x),1)
y=[4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,5.99607266,12.46321171,47.70487443,65.70793999,62.7767844 ,35.22558438,77.84563303,11.08106882]
y=np.array(y)
y_new=np.log(y)

x4=np.hstack((np.ones((20,1)),x, x**2))
reg_res = normal_equations(x4,y_new)
print(np.exp(reg_res[0]),reg_res[1],reg_res[2])