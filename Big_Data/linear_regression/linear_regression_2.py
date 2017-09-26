import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#retrive data
mnist = pd.read_csv("MNIST_15_15.csv", header = None)
mnist_label = pd.read_csv("MNIST_LABEL.csv", header = None)

#shuffle the data index for extracting the data later
data_index = np.arange(mnist.shape[0])
np.random.shuffle(data_index)

#split data index into 10 batches for 10 CV
batch = []
for i in range (10):
    batch.append([])
    for j in range (len(data_index)):
        if j%10 == i:
            batch[i].append(data_index[j])

#loop for each batch
for i in range(len(batch)):
    #X is the train data, y is the labels
    X = mnist.__deepcopy__()
    y = mnist_label.__deepcopy__()
    #drop the selected set of data index(test data) from X and add a column of intercept
    X = np.array(X.drop(batch[i]))
    ones = np.ones((X.shape[0],1))
    X = np.hstack((X,ones))
    #drop the selected set of data index(test data) from the label, also change label, 5 to -1, 6 to 1
    y = np.array(y.drop(batch[i]))*2-11
    std = np.std(X,axis = 1)

    ###standard deviation normalization by column
    std =np.std(X, axis = 0)
    mean = np.mean(X, axis = 0)
    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            if std[k] > 0:
                X[j][k] = (X[j][k]-mean[k])/std[k]

    #find the coefficient
    b = ((np.linalg.inv(np.transpose(X).dot(X)+0.1*np.identity(X.shape[1]))).dot(np.transpose(X))).dot(y)

    #retrieve the test data
    test_X =[]
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

    #make the attual answer and prediction from matrix 34*1 to 1*34
    test_y = test_y.flatten()
    prediction = prediction.flatten()

    #find the correct rate, if the prediction < 0, it is 5, if prediction > 0, it is 6
    correct_count = 0
    for j in range(len(prediction)):
        if prediction[j] > 0 and test_y[j]==6:
            correct_count+=1
        elif prediction[j] < 0 and test_y[j]==5:
            correct_count+=1
        else:
            pass
    print ("iteration", i, "accuracy:",correct_count/len(test_y))

    '''
        #standard deviation normalization by row
        std =np.std(X, axis = 1)
        for j in range(X.shape[0]):
            mean = sum(X[j])/X.shape[1]
            row_std = std[j]
            for k in range(X.shape[1]):
                if row_std > 0:
                    X[j][k] = (X[j][k]-mean)/row_std

         ###standard deviation normalization by whole matrix
        std =np.std(X)
        mean = np.mean(X)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                if std > 0:
                    X[j][k] = (X[j][k]-mean)/std
    '''