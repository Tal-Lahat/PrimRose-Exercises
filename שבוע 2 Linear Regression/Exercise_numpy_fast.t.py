#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 
# Welcome to Numpy a package widely used in the data science community. 
# In this assignment, you will:
# * Learn to use Numpy, a package which let us work efficiently with arrays and matrices in Python.
# 
# 
# #### Why are we using Numpy? 
# 
# * Extra features required by Python :
# * fast, multidimensional arrays
# * libraries of reliable, tested scientiﬁc functions
# * plotting tools
# * NumPy is at the core of nearly every scientific Python application or module since it provides a fast N-d array datatype that can be manipulated in a vectorized form. 

# ## Load package
# * Let's import Numpy as np. This lets us use the shortcut np to refer to Numpy..





import numpy as np

# * T1- Check the numpy version ד
#       Hint: methods version.version or __version__


# **Our first Numpy array. We can start by creating a list and converting in to an array**: 


mylist = [1, 2, 3]
x = np.array(mylist)
print(x)
##

# ** x is our first Numpy array. We can pass the list directly**:

y = np.array([4, 5, 6])
print(y)
#%%

# **Multidimensional arrays by passing in a list of lists.
# (We passed in two lists with three elements each, and we get a two by three array.)**: 
# 
# * T2- Create a numpy array m  with the vectors: [7, 8, 9] and  [10, 11, 12]
# * T3- Check the dimensions by using the shape attribute
# 
m = np.array([[7,8,9],[10,11,12]])
print("m: ", m)
#Check the dimensions by using the shape attribute:
print(m.shape)

"""Expected Output:
m:  [[ 7  8  9]
 [10 11 12]]
(2, 3)
"""
#%%

# **Number of rows of array**: 
m = np.array([[7, 8, 9], [10, 11, 12]])
# T4. check the number of rows with the len function:
print(len(m))

"""Expected Output: 2"""


# T5. check the number of rows with the shape attribute:
print(m.shape[0])

"""Expected Output: 2"""


# ### 1 - arange function:
# We pass in a start, a stop, and a step size, 
# and it returns evenly spaced values within a given interval.
# %%
# * T6. Create a numpy array n that start at 0 count up by 2 and stop before 30
# 
n = np.arange(0,30,2)
print(n)
"""Expected Output: x, y = np.mgrid[0:5, 0:5]
   array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
"""

#%%
# ### 2- reshape method:
# Suppose we wanted to convert this array of numbers to a three by five array. 
# We can use reshape to do that.:
# 
# * T7. Reshape array to be 3x5
n = n.reshape(3,5)
display(n)

""" Expected output:
array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28]])
"""

#%%
# ### 3- linspace function:
# It is similar to arange, except we tell it how many numbers we want returned, and it will split up the interval according:
# 
# * T8. Create numpy array o returning 9 evenly spaced values from 0 to 4
o = np.linspace(0, 4, 9)
display(o)

""" Expected output:
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. ])
"""
#%%

# * T9. Now please resize to change the dimensions in place to 3x3:
display(o.reshape(3,3))
""" Expected output
  array([[0. , 0.5, 1. ],
       [1.5, 2. , 2.5],
       [3. , 3.5, 4. ]])
"""

#%%
# ## Built-in functions and shortcuts for creating arrays.
# 
# 

# ### 1- ones function:
# Returns an array of ones
# * T10. Create an array of ones with dimensions 3x2 and after multiply per 10
n=10*np.ones((3,2))
display(n)

""" Expected output
  array([[10., 10.],
       [10., 10.],
       [10., 10.]])
"""
#%%

# ### 2- zeros function:
# Returns an array of zeros
# 
# * T11. Create an array of zeros with dimensions 2x3
n=np.zeros((2,3))
display(n)
""" Expected output
array([[0., 0., 0.],
       [0., 0., 0.]])
"""
#%%

# ### 3- eye function:
# Returns an array with ones on the diagonal and zeros everywhere else
# 
# * T12. Create an identity matrix with dimensions 3x3
n=np.eye((3))
display(n)
""" Expected output
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
"""
#%%

# ### 4- diag function:
# Constructs a diagonal array
# 
mylist = [1, 2, 3] # just putting this here so x and y are defined
x = np.array(mylist)
y = np.array([4, 5, 6])

# * T13. Create diagonal array based on previous y array 
print ("y:" ,y)
display(np.diag(y))
""" Expected output
y: [4 5 6]
array([[4, 0, 0],
       [0, 5, 0],
       [0, 0, 6]])
"""
#%%


# ### 5- repeat function:
# Return an array with repeated values
# 
# * T14. Repeat the array [1,2,3] three times.
display(np.repeat([1,2,3],3))
""" Expected output
array([1, 1, 1, 2, 2, 2, 3, 3, 3])
"""

#%%
# Notice the difference when we pass in a repeated list:
# 
# * T15. Multiply the list [1,2,3] per 3 and convert it in a numpy array.
n=np.array([1,2,3]*3)
display(n)
# l=[1,2,3]*3

# display(l)
""" Expected output
array([1, 2, 3, 1, 2, 3, 1, 2, 3])
"""

#%%
# 
# ### 6- vstack function:
# Return a array stack it vertically
# 
# *  T16. Stack the array p vertically with itself multiplied by 2.
p = np.ones([2, 3], int) 
# 2*p
display(np.vstack((p,2*p)))
"""Insert code here"""
""" Expected output
array([[1, 1, 1],
       [1, 1, 1],
       [2, 2, 2],
       [2, 2, 2]])
"""
#%%

# ### 7- hstack function:
# Return an array stack it horizontally.
# 
# * T17. Stack the previous array p horizontally with itself multiplied by 2.
display(np.hstack((p,2*p)))
""" Expected output
array([[1, 1, 1, 2, 2, 2],
       [1, 1, 1, 2, 2, 2]])
"""

#%%
# ## Operations.
# 
# 

# ### NumPy operations are usually done on pairs of arrays on an element-by-element basis:
# 
# Hint: From previous exercises:
# 
x =  np.array([1 ,2 ,3])
y =  np.array([4, 5, 6])
# display(x+y)
# ### elementwise addition:
# 
# * T18. Calculate and print the result:   [1 2 3] + [4 5 6] = [5  7  9]
print(np.array([1,2,3])+np.array([4,5,6]))
""" Expected output
  [5 7 9]
"""

#%%
# 
# ### elementwise substraction:
# 
# * T19. Calculate and print the result:   [1 2 3] - [4 5 6] = [-3 -3 -3]
print(np.array([1,2,3])-np.array([4,5,6]))
""" Expected output
  [-3 -3 -3]
"""

#%%
# 
# ### elementwise multiplication:
# 
# * T20. Calculate and print the result:   [1 2 3] * [4 5 6] = [4  10  18]
print(np.array([1,2,3])*np.array([4,5,6]))

""" Expected output
  [ 4 10 18]
"""
#%%

# 
# ### elementwise division:
# 
# * T21. Calculate and print the result: [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
print(np.array([1,2,3])/np.array([4,5,6]))

""" Expected output
  [0.25 0.4  0.5 ]
"""
#%%

# 
# ### elementwise power:
# 
# *  T22. Calculate and print the result: [1 2 3] ^2 =  [1 4 9]
print(np.array([1,2,3])**2)

""" Expected output
  [1 4 9]
"""
#%%


# 
# ### dot product function (linear algebra):
# 
# * T23. Calculate and print the dot product between x and y:  1x4 + 2x5 + 3x6
print(np.dot(x,y))
""" Expected output 32 """

#%%
# 
# ###  Transpose of an array using the t method: 
# Swaps the rows and columns.
# 
# * T24. Transpose z
z = np.array([y, y**2]).T
display(z)

z.shape

"""Insert code here"""

""" Expected output
array([[ 4, 16],
       [ 5, 25],
       [ 6, 36]])
"""
#%%

# 
# ####  transpose function: 
# 
# * T25. Now use the np.transpose function with z
# 
#np.transpose function:
z = np.array([y, y**2])
display(np.transpose(z))

"""Insert code here"""

""" Expected output
array([[ 4, 16],
       [ 5, 25],
       [ 6, 36]])
"""
#%%

# ### dtype class:
# Return type of the data (integer, float, Python object, etc.)
# 
# * T26. Show z type of data
# 
"""Insert code here"""
display(z.dtype)
""" Expected output
  dtype('int32')
"""
#%%

# T27. with astype, cast an array to a different type. ; float.  
z= z.astype('float32')
display(z.dtype)

""" Expected output
  dtype('float32')
"""

#%%
# ## Math functions.
# 
# NumPy has quite a few useful  mathematical and statistical functions for finding minimum: min, maximum: max, standard deviation: std , sum, mean, indexes of maximum and minimum value: argmax and argmin, etc. from the given elements in the array.
# 
# * Calculate the required functions in each comment:
# 
# Hint:  a = np.array([-4, -2, 1, 3, 5])
# 
a = np.array([-4, -2, 1, 3, 5])

# T28. sum of the values of the array
print(a.sum())
""" Expected output 3 """


# T29. maximum of the values of the array
"""Insert code here"""
print(a.max())
""" Expected output 5 """


# T30. minimum of the values of the array
"""Insert code here"""
print(a.min())
""" Expected output -4"""


# T31.mean of the values of the array
"""Insert code here"""
print(a.mean())
""" Expected output 0.6 """


# T32. standard deviation of the values of the array
"""Insert code here"""
print(a.std())
""" Expected output 3.2619012860600183"""


# T33. argmax: index of a maximum value

"""Insert code here"""
print(a.argmax())
""" Expected output 4 """


# T34. argmin: index of the minimum value
"""Insert code here"""
print(a.argmin())
""" Expected output 0 """

#%%
# ## Indexing and slicing
# 
# 

# We can use bracket notation to get the value at a particular index, 
# and the colon notation to get a range.
# The notation is start and stepsize.
# Specifying the starting or ending index is not necessary. 

s = np.arange(13)**2 #array with the squares of 0 through 12. 
print("s: " ,s)

print((s[0], s[4], s[-1]))


""" Expected output
 s:  [  0   1   4   9  16  25  36  49  64  81 100 121 144]
 (0, 16, 144)
"""
#%%

# * T35. For our first example (s array), Take a look at the range starting from index one and 
# stopping before index five
"""Insert code here"""
display((s[1:5]))
""" Expected output
array([ 1,  4,  9, 16], dtype=int32)
"""
#%%

# We can also use negatives to count back from the end of the array. 
# 
# * T36. Get a slice of the last four elements of the array.
"""Insert code here"""
display((s[-4:]))
""" Expected output
array([ 81, 100, 121, 144], dtype=int32)
"""
#%%
# T37 Starting fifth from the end to the beginning of the array and counting backwards by two.
# Remember s:[  0   1   4   9  16  25  36  49  64  81 100 121 144]
"""Insert code here"""
display((s[-5:0:-2]))
""" Expected output
array([64, 36, 16,  4,  0], dtype=int32)
"""
#%%

# When the step value is negative. In this case, the defaults for start and stop are swapped. This becomes a convenient way to reverse an array:
# 
# * T38. Reverse the array s
"""Insert code here"""
display((s[::-1]))
""" Expected output
array([144, 121, 100,  81,  64,  49,  36,  25,  16,   9,   4,   1,   0],
      dtype=int32)
"""
#%%
r = np.arange(36)
r.resize((6, 6)) # two dimensional array, 0 to 35. 
display(r)

""" Expected output
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
"""
#%%

# * T39. Use colon notation to get a slice of the third row and 
#   columns three to six. 
"""Insert code here"""
# r1=r[0:2][2:6]
# display(r1)
display(r[2,2:6])
# display(r[2][2:5])
""" Expected output
array([14, 15, 16, 17])
"""
#%%

# * T40. We can also do something like get the first two rows and all of the columns except the last.
"""Insert code here"""
# display(r[0][0:5],r[1][0:5])
display(r[0:2,0:5])
""" Expected output
array([[ 0,  1,  2,  3,  4],
       [ 6,  7,  8,  9, 10]])
"""

#%%
# * T41. Select every second element from the last row.
"""Insert code here"""
display(r[-1,0::2])
""" Expected output
array([30, 32, 34])
"""
#%%

# We can also use the bracket operator to do conditional indexing and assignment.
# 
# * T42. Select elements of array r that are greater than 30.
"""Insert code here"""
display(r[r>30])
""" Expected output
array([31, 32, 33, 34, 35])
"""
#%%

# We can also use the bracket operator to do conditional indexing and assignment.
# 
# * T43. Take elements of array r that are greater than 30 and assign them to the new value = 30.
"""Insert code here"""
r[r>30]=30
display(r)
""" Expected output
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])
"""
#%%

# In numpy the copy is by reference not by value, what does it mean? when you copy an array to another variable the changes in this variable affect the original array.
# Example: 
# First, let's create a new array r2, which is a slice of the array r. Now, let's set all the elements of this array to zero.
r2 = r[:3,:3]
r2

r2[:] = 0
r2

r # When we look at the original array r,  we can see that the slice in r has also been changed. 

""" Expected output
array([[ 0,  0,  0,  3,  4,  5],
       [ 0,  0,  0,  9, 10, 11],
       [ 0,  0,  0, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])
"""
#%%

# If we wish to create a copy of the r array that will not change the original array, 
# we can use the copy method:
# 
# * T44. Use the copy method from the r array to r_copy:
r_copy = r.copy()
display(r_copy)

""" Expected output
array([[ 0,  0,  0,  3,  4,  5],
       [ 0,  0,  0,  9, 10, 11],
       [ 0,  0,  0, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 30, 30, 30, 30, 30]])
"""
#%%

# We can see that if we change the values of all the elements in r_copy to 5, 
# the original array r remains unchanged.
# 
r_copy[:] = 5
print(r_copy, '\n')
print(r)

""" Expected output
[[5 5 5 5 5 5]
 [5 5 5 5 5 5]
 [5 5 5 5 5 5]
 [5 5 5 5 5 5]
 [5 5 5 5 5 5]
 [5 5 5 5 5 5]] 

[[ 0  0  0  3  4  5]
 [ 0  0  0  9 10 11]
 [ 0  0  0 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 30 30 30 30 30]]
"""

#%%
# ### Indexing
# 
# #### 1- nonzero function:
# Return the indices of the elements that are non-zero.
# 
# * T45. Find and print indices of non-zero elements from [3,1,0,0,5,0]
nz = [3,1,0,0,5,0]

print(np.nonzero(nz))

""" Expected output
(array([0, 1, 4], dtype=int64),)
"""
#%%

# 
# #### 2- Boolean or “mask” index arrays:
# With boolean indexing, we can use an array of boolean values to subset another array. 
# For example, four2D is a 4x4 array and we’d like to replace every 5 with 0.
# Running foo == 5 gives us a 4x4 array of boolean values which we’ll store in a variable called my_mask.
# Now we can use this array of boolean values (my_mask) to index our original array,identifying which elements are 5, and setting them equal to 0.
# 
# 4x4 array of positive integers.
four2D = np.array([
    [13, 5, 7, 5],
    [5, 0, 3, 5],
    [3, 5, 5, 0],
    [5, 4, 5, 3]
])

# Checking four2D == 5, numpy gives us a 4x4 array of boolean values
my_mask = four2D == 5

print(my_mask)

""" Expected output
[[False  True False  True]
 [ True False False  True]
 [False  True  True False]
 [ True False  True False]]
"""
#%%

# * T46. Use this array of boolean values to index four2D array and setting 5 values to 0:
four2D = np.array([
    [13, 5, 7, 5],
    [5, 0, 3, 5],
    [3, 5, 5, 0],
    [5, 4, 5, 3]
])
# four2D[four2D==5]=0
four2D[my_mask==True]=0
display(four2D)

""" Expected output
array([[13,  0,  7,  0],
       [ 0,  0,  3,  0],
       [ 3,  0,  0,  0],
       [ 0,  4,  0,  3]])
"""
#%%

# * T47. Use 1d boolean arrays row_2_3 and col_1_4 to pick out 2nd and 3rd rows or 1st and 4th columns
col_1_4 = np.array([True, False, False, True])
row_2_3 = np.array([False, True, True, False])
a=four2D[row_2_3==True]  # returns rows 2 and 3
print(a)
b=four2D[:,col_1_4==True]  # returns cols 1 and 4
print(b)

""" Expected output
[[0 0 3 0]
 [3 0 0 0]]
[[13  0]
 [ 0  0]
 [ 3  0]
 [ 0  3]]
"""

#%%
# ####  3- Logical operators and logical functions
# 
# Logical operators let us combine boolean arrays. They include
# 
# the bitwise "and" operator: &  we can use np.logical_and() instead.
# 
# the bitwise "or" operator: |  we can use np.logical_or() instead.
# 
# And the bitwise "xor" operator: ^  we can use np.logical_xor() instead.
# 
# We can also negate a boolean array by preceding it with a tilde: "~"  we can use np.logical_not() instead.
# 
# We have an array of proper nouns and corresponding arrays for their age and gender. 
# 
# * T48. Who is at least 24?
# * T49. Which females are over 30?
# * T50. Who is a not a female or younger than 40?
names = np.array(["Sarah", "Elad", "Alan", "Elena", "Hana", "Ilan"])
ages = np.array([24, 12, 43, 31, 65, 20])
genders = np.array(['female', 'male', 'male', 'female', 'female', 'male'])

#Who is at least 24?
a= names[ages>=24]
print(a)
#%%
# Which females are over 30? לשאול
# a= names[genders=='female' and ages>30 ]
a= names[np.logical_and(genders=='female',ages>30)]
print(a)
#%%
# Who is a not a female or younger than 40?
a= names[np.logical_or(ages<40,np.logical_not(genders=='female'))]
print(a)
""" Expected output
['Sarah' 'Alan' 'Elena' 'Hana']
['Elena' 'Hana']
['Sarah' 'Elad' 'Alan' 'Elena' 'Ilan'] """
#%%
# ####  4- where (condition[, x, y])
# Return elements chosen from x or y depending on condition.
# 
# * T51. Select the indexes corresponding to a slice that matches a criteria such as all the value a1[:,1] included in the range (l1,l2):
a1 = np.zeros( (10,2) )

a1[:,0]=np.arange(0,10)
a1[:,1]=np.arange(0.5,20,2)
print(a1)
l1=2.0; l2=18.0; 
result= np.where(np.logical_and(a1[:,1]>l1, a1[:,1]<l2))
print(result)
""" Expected output
[[ 0.   0.5]
 [ 1.   2.5]
 [ 2.   4.5]
 [ 3.   6.5]
 [ 4.   8.5]
 [ 5.  10.5]
 [ 6.  12.5]
 [ 7.  14.5]
 [ 8.  16.5]
 [ 9.  18.5]]
(array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int64),)
"""
#%%
# 
# ## Iterate over arrays.
# 
# We can iterate by row by typing for row in test, for example.

test = np.random.randint(0, 10, (4,3)) #four by three array of random numbers, from zero through nine
test

""" Expected output
array([[0, 4, 1],
       [2, 3, 9],
       [9, 4, 2],
       [5, 4, 4]])
"""
# count=0
for row in test:
    print(row)
#     count=count+1
# print (count)
    
""" Expected output
[0 4 1]
[2 3 9]
[9 4 2]
[5 4 4]
"""
# for row in test:
#     for val in row:
#         print(val)
#%%

# * T52 .We can iterate by row index by using the length function on test, 
# which returns the number of rows.
# 
for i in range(len(test)):
    print (str(i) + ": " + str(test[i]))
    
""" Expected output
0: [0 4 1]
1: [2 3 9]
2: [9 4 2]
3: [5 4 4]
"""
#%%

# 
# We can combine these two ways of iterating by using enumerate, 
# which gives us the row and the index of the row.
# 
# * T53. Make a new array, test2 using enumerate() with the test variable
test2= """Insert code here"""
for i, row in test2:
    print('row', i, 'is', row)

""" Expected output
row 0 is [0 4 1]
row 1 is [2 3 9]
row 2 is [9 4 2]
row 3 is [5 4 4]
"""

#%%
# 
# If we wish to iterate through both arrays, we can use zip.
# 
# * T54. Use zip with test and test2
test2 = test**2
for i, j in zip(test,test2):
    print(i,'+',j,'=',i+j)

""" Expected output
[0 4 1] + [ 0 16  1] = [ 0 20  2]
[2 3 9] + [ 4  9 81] = [ 6 12 90]
[9 4 2] + [81 16  4] = [90 20  6]
[5 4 4] + [25 16 16] = [30 20 20]
"""
#%%

# 
# #### nditer() function
# 
# nditer is used for Iterating on Each Scalar Element
# 
# * T55. Iterate through the following 3-D array:
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr)

for x in """Insert code here"""
  print(x)

""" Expected output
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
1
2
3
4
5
6
7
8
"""


# To iterate two arrays simultaneously, we can pass two arrays to the nditer function. 
# * T56. Iterate the arrays ‘A’ and ‘R’ simultaneously:
A= np.arange(12).reshape(4,3)
R = np.arange(3)

for a,r in #Insesrt code here
    print(a, r)

""" Expected output
0 0
1 1
2 2
3 0
4 1
5 2
6 0
7 1
8 2
9 0
10 1
11 2
"""


# 
# #### nindex() function
# 
# An N-dimensional iterator object to index arrays.
# 
# Given the shape of an array, an ndindex instance iterates over the N-dimensional index of the array. 
# 
# * T57. Iterate through the following 2-D array:
A= np.arange(12).reshape(4,3)

for ix,iy in """Insert code here"""
    print(A[ix,iy])

""" Expected output
0
1
2
3
4
5
6
7
8
9
10
11
"""


# #### ndenumerate() function
# 
# Return an iterator yielding pairs of array coordinates and values.
# 
# * T58. Print coordinates and values:
a =np.array([[1,2],[3,4],[5,6]])
for (x,y), value in """Insert code here"""
    print (x,y, value)

""" Expected output
0 0 1
0 1 2
1 0 3
1 1 4
2 0 5
2 1 6
"""
    

#%%
# 
# ## File I/O.
# 
# Save in a text file and a numpy file format:
# 
# * T59. Save test2 in a file called datasaved.npy using the function save.
np.savetxt('datasaved.txt', test2)

#%%

#load the file datasaved.npy in the variable test_from_file and print it.

test_from_file = np.loadtxt('datasaved.txt')
print(test_from_file)

""" Expected output
[[ 0 16  1]
 [ 4  9 81]
 [81 16  4]
 [25 16 16]]
"""
#%%

# T60. Save test in a file called datasaved.txt using the function savetxt
"""Insert code here"""


# T61. Use the function genfromtxt to load the file datasaved.txt in the variable data and print it.
#      delimiter= ' ' and skip_header= 0
data = """Insert code here"""
print(data)

""" Expected output
[[0. 4. 1.]
 [2. 3. 9.]
 [9. 4. 2.]
 [5. 4. 4.]]
"""


# 
# ## Sorting
# 
# Acts in array himself (method):
# 
# * T62. Sort the array data using the method sort 
"""Insert code here"""
data

""" Expected output
array([[0., 1., 4.],
       [2., 3., 9.],
       [2., 4., 9.],
       [4., 4., 5.]])
"""


# 
# Numpy function sort:
# * T63. Sort the array data using the numpy function
"""Insert code here""" 

""" Expected output
array([[0., 1., 4.],
       [2., 3., 9.],
       [2., 4., 9.],
       [4., 4., 5.]])
"""


# 
# Numpy function argsort:
# 
# * T64. Return the indices that would sort the data array. 
s= data."""Insert code here"""
s

""" Expected output
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]], dtype=int64)
"""


# * T65. Return the indices that would sort the data array in descend order 
# * Hint: Descend order: Negate the array
s= """Insert code here"""
s

""" Expected output
array([[2, 1, 0],
       [2, 1, 0],
       [2, 1, 0],
       [2, 0, 1]], dtype=int64)
"""


# T66. Order data array in descend order 
result= """Insert code here"""
result

""" Expected output
array([[4., 1., 0.],
       [9., 3., 2.],
       [9., 4., 2.],
       [5., 4., 4.]])
"""





