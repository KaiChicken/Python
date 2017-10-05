"""
Deopendencies:
tensorflow: 1.1.0
matplotlib
numpy
pandas
run command line: python gender_classification_CNN.py
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#retrieve data from wiki5.csv as matrix
data = pd.read_csv("wiki5.csv", index_col=0)
x_data = data.drop(['gender', 'age'], axis=1)
y_data = data['gender']

x_data = np.array(x_data,dtype=float)/255
y_data = np.array(y_data,dtype=float)
y_data = np.array(y_data)[:,np.newaxis]
y_data = np.insert(y_data, 1, 1 ,axis=1)
for i in y_data:
    if i[0] == 0:
        i[0] = 1
        i[1] = 0
    else:
        i[0] = 0
        i[1] = 1


#retrieve data from imdb4p5.csv as matrix
data_2 = pd.read_csv("imdb4p5.csv", index_col = 0)
x_data_2 = data_2.drop(['gender', 'age'], axis = 1)
y_data_2 = data_2['gender']

x_data_2 = np.array(x_data_2,dtype=float)/255
y_data_2 = np.array(y_data_2,dtype=float)
y_data_2 = np.array(y_data_2)[:,np.newaxis]
y_data_2 = np.insert(y_data_2, 1, 1 ,axis=1)
for i in y_data_2:
    if i[0] == 0:
        i[0] = 1
        i[1] = 0
    else:
        i[0] = 0
        i[1] = 1
#get the first 2000 image as test data from the imdb4p5.csv file
x_test = x_data[:2000]
y_test = y_data[:2000]
#print(x_test.shape)
#print(y_test.shape)

#show the image
#plt.imshow(x_data[0].reshape((100, 100)), cmap='gray')
#plt.title('%i' % np.argmax(y_data[0]))
#plt.show()

#set the placeholder of the CNN
tf_x = tf.placeholder(tf.float32, [None, 10000])
image = tf.reshape(tf_x, [-1,100,100,1])
tf_y = tf.placeholder(tf.int32, [None, 2])
tf_is_training = tf.placeholder(tf.bool, None)

#CNN
#convert to 100*100*64
#create feature maps of 100*100*64 with 3*3
conv1 = tf.layers.conv2d(
    inputs = image,
    filters = 64,
    kernel_size = 3,
    strides = 1,
    padding = 'same',
    activation = tf.nn.relu
)
#convert to 50*50*64
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size = 2,
    strides = 2,
)
#create feature maps of 50*50*128 with 3*3
conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)
#convert to 25*25*128
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
#print('pool2', pool2.shape)

#create feature maps of 25*25*256
conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)
#convert to 24*24*256
pool3 = tf.layers.max_pooling2d(conv3, 2, 1)
#print('pool3', pool3.shape)

#create feature maps of 24*24*512
conv4 = tf.layers.conv2d(pool3, 512, 3, 1, 'same', activation=tf.nn.relu)
#convert to 12*12*512
pool4 = tf.layers.max_pooling2d(conv4, 2, 2)


flat = tf.reshape(pool4, [-1, 12*12*512])
l1 = tf.layers.dense(flat, 2048, activation=tf.nn.relu)
#l1 = tf.layers.dropout(l1, rate=0.2,training=tf_is_training)
l2 = tf.layers.dense(l1, 2048, activation=tf.nn.relu)
#l2 = tf.layers.dropout(l2, rate=0.2,training=tf_is_training)
l3 = tf.layers.dense(l2, 1000, activation=tf.nn.relu)
#l3 = tf.layers.dropout(l3, rate=0.2,training=tf_is_training)
output = tf.layers.dense(l3, 2)

#calculate loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
#train the model to minimize loss using adam optimizer
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
#find out the accuracy of the training data
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis =1),predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

for step in range(100):
    start = step*330
    end = step*330+330
    b_x = x_data_2[start:end]
    b_y = y_data_2[start:end]
    print(start,end)
    loss_, _  = sess.run([loss, train_op], {tf_x:b_x, tf_y:b_y, tf_is_training:False})
    if step%10 == 0:
        print('| train loss: %.4f' % loss_)
    #    acc, flat_representation = sess.run([accuracy, flat], {tf_x: x_data, tf_y: y_data})
    #    print('step', step, '| train loss: %.4f' % loss_, '| accuracy: %.2f' % acc)


#test the model with first 2000 images in wiki5.csv
print('Testing the model... ')
test_output = sess.run(output, {tf_x: x_test, tf_is_training:False})
predicted_output = np.argmax(test_output,1)
ground_truth = np.argmax(y_test, 1)
correct_number = sum(np.equal(predicted_output, ground_truth))
print(correct_number)
print("correct rate of 2000 image: %.4f" % (correct_number/2000))

