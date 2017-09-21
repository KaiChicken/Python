import numpy as np
import time

current_time = time.time()
x_data = np.random.rand(100).astype(np.float32)
print (x_data)
a = np.random.rand(5)
print (a)
print (a[0])

array = np.array([[1,2,3],
                 [2,3,4],
                 [3,4,5]]
                 )
print (array)
print ('number of the dimension:',array.ndim)
print ('shape:', array.shape)
print ('size:', array.size)

a = np.zeros((3, 4))

a = np.ones((3, 4))
print(a)
print(a.dtype)

for x in range(10,20,2):
    print(x)

a = np.arange(10,20,2)
print(a)

a = np.arange(12).reshape((3,4))
print(a)

a = np.linspace(0,10,6).reshape((2,3))
print(a)


### calculation
print("")
print("calculation")
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)
c = a-b
print(c)

c = 10*np.sin(a)
print(c)

print(b)
print(b<3)

a = np.array([[1,1],
             [0,1]])
b = np.arange(4).reshape((2,2))
c = a*b
c_dot = np.dot(a,b)
c_dot_2 = a.dot(b)
print(c)
print(c_dot)
print(c_dot_2)

#random number
a = np.random.random((2,4))
print(a)
print(np.sum(a))
print(np.min(a))
print(np.max(a))

#every column
b = np.sum(a, axis = 0)
print(b)
b = np.min(a, axis = 1)
print(b)

A = np.arange(2,14).reshape((3,4))
#index of minimum number
print(np.argmin(A))
#index of max number in a line
print(np.argmax(A))
print(np.mean(A))
print(A.mean())
print(np.average(A))
#median vs mean
print(np.median(A))

#cumulative sum, every number is sum of one/all previous number and current number
print (A)
print(np.cumsum(A))

#the different between previous and current number, size would change
print(A)
print(np.diff(A))

#find non zero number, give x and y index of nonzero number, first is x axis?
print(A)
print(np.nonzero(A))

#sort for every row
A = np.arange(14,2,-1).reshape((3,4))
print(A)
print(np.sort(A))

#transpose
A = np.arange(14,2,-1).reshape((3,4))
print(np.transpose(A))
print(A.T.dot(A))

#if less the first number, it becomes the first number, if greater than second numberm, it becomes the second numebr
print(A)
print(np.clip(A,5,9))
print(np.mean(A, axis = 0))

A = np.arange(3,15).reshape(3,4)
print(A)
print(A[2])

print (A)
print(A.T)
for column in A.T:
    print(column)
print(A.flatten())
for item in A.flat:
    print(item)

A = np.array([1,1,1])
B = np.array([2,2,2])
#vertical stack
C = np.vstack((A,B))
print(A.shape, C.shape)
print(C)
#horizontal stack
D = np.hstack((A,B))
print(D)

#Transpose doesnt work for one row only
print(A.T)

print(A[np.newaxis,:].shape)
print(A[:,np.newaxis].shape)
print(A[:,np.newaxis])

#make one row array as transpose
A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]
C = np.vstack((A,B))
D = np.hstack((A,B))
print(C)
print(D)
#axis 0 = verticl, axis 1 = horizontal
C = np.concatenate((A,B,B,A), axis = 1)
print(C)

#extract matrix...  to be continued



#copy and deep copy
a = np.arange(4)
print(a)
b = a
c = a
d = b
#deep copy
b = a.copy()
a[0] = 11
print(a,b,c,d)
#test if they are same
print (b is a)

print(time.time()-current_time)

import pandas as pd
pd_data = pd.read_csv("housing_training.csv",header=None)
print(pd_data.iloc[0])
np_data = np.array(pd_data)
print(np_data[0])