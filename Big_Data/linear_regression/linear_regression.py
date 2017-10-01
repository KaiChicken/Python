import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#retrieve housing_training data
housing_training = pd.read_csv("housing_training.csv",header=None)
#split the data as X = variables, y = answer
X = housing_training.__deepcopy__()
#add a column of 1s at the end of X, so that it will give the intercept value
X.loc[:,13] = 1
X = np.array(X)
y = np.array(housing_training.loc[:,13])[:,np.newaxis]
#calculate the coefficients
b = ((np.linalg.inv(np.transpose(X).dot(X))).dot(np.transpose(X))).dot(y)
print("coefficients are")
print(b)

#retrieve housing_test data, make the last column of test_X = 1, test_y is the actual result
housing_test = pd.read_csv("housing_test.csv",header=None)
test_X = housing_test.__deepcopy__()
test_X.loc[:,13] = 1
test_X = np.array(test_X)
test_y = np.array(housing_test.loc[:,13])
#find predictions using the b from training set
prediction = (test_X.dot(b)).flatten()

#find RMSE
rmse = (((sum((test_y-prediction)**2))/test_y.shape[0])**0.5)
print("RMSE is", rmse)

#plot the graph
x=[]
for i in range(100):
    x.append(i)
plt.plot(x,color = 'red')
plt.scatter(prediction,test_y,color = 'black')
plt.xlabel('Prediction')
plt.ylabel('Ground Truth')
plt.xlim(0,52)
plt.ylim(0,52)
plt.show()