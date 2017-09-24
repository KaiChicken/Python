import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#retrive data
mnist = pd.read_csv("MNIST_15_15.csv", header = None)
mnist_label = pd.read_csv("MNIST_LABEL.csv", header = None)

#split data into 10 batches
batch = []
for i in range (10):
    batch.append([])
    for j in range (mnist.shape[0]):
        if j%10 == i:
            batch[i].append(j)

#loop for each batch
for i in range(len(batch)):
    #X is the train data, y is the labels
    X = mnist.__deepcopy__()
    y = mnist_label.__deepcopy__()
    #drop the test set from X and add a column of intercept
    X = np.array(X.drop(batch[i]))+0.0001
    ones = np.ones((X.shape[0],1))
    X = np.hstack((X,ones))
    #drop the test set from the label, also change label, 5 to -1, 6 to 1
    y = np.array(y.drop(batch[i]))*2-11
    std = np.std(X,axis = 1)

    #since the 0s in the matrix makes the matrix not invertible, i add a very small value to the 0s
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            X[j][k] = X[j][k]+ np.random.random_sample()/10000

    #standard deviation normalization
    std =np.std(X, axis = 1)
    for j in range(X.shape[0]):
        mean = sum(X[j])/X.shape[1]
        row_std = std[j]
        for k in range(X.shape[1]):
            if row_std > 0:
                X[j][k] = (X[j][k]-mean)/row_std

    #find the coefficient
    b = ((np.linalg.inv(np.transpose(X).dot(X))).dot(np.transpose(X))).dot(y)

    #retrieve the test data
    test_X = []
    test_y =[]
    for j in range (len(batch[i])):
        test_X.append(mnist.loc[batch[i][j]])
        test_y.append(mnist_label.loc[batch[i][j]])
    test_X = np.array(test_X)
    ones = np.ones((test_X.shape[0],1))
    test_X = np.hstack((test_X,ones))
    #retrieve the actual answers
    test_y = np.array(test_y)
    #calculate prediction
    prediction = test_X.dot(b)

    #make the matrix from 34*1 to 1*34
    test_y = test_y.flatten()
    prediction = prediction.flatten()

    #find the correct rate
    correct_count = 0
    for j in range(len(prediction)):
        if prediction[j] > 0 and test_y[j]==6:
            correct_count+=1
        elif prediction[j] < 0 and test_y[j]==5:
            correct_count+=1
        else:
            pass
    print ("iteration", i, "accuracy:",correct_count/len(test_y))