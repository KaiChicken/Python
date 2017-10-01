"""
Deopendencies:
tensorflow: 1.1.0
matplotlib
numpy
pandas
python gender_classification_CNN.py
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("wiki5.csv", index_col=0)
x_data = data.drop(['gender', 'age'], axis=1)
y_data = data['gender']
x_data = np.array(x_data,dtype=float)
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
a = x_data[0:50]
print(a.shape)
#plt.imshow(x_data[0].reshape((100, 100)), cmap='gray')
#plt.title('%i' % np.argmax(y_data[0])); plt.show()

tf_x = tf.placeholder(tf.float32, [None, 10000])/255
image = tf.reshape(tf_x, [-1,100,100,1])
tf_y = tf.placeholder(tf.int32, [None, 2])

#CNN
conv1 = tf.layers.conv2d(
    inputs = image,
    filters = 16,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size = 2,
    strides = 2,
)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
flat = tf.reshape(pool2, [-1, 25*25*32])
output = tf.layers.dense(flat,2)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis =1),predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

for step in range(80):
    start = step*50
    end = step*50+50
    b_x = x_data[start:end]
    b_y = y_data[start:end]
    _, loss_ = sess.run([train_op, loss], {tf_x:b_x, tf_y:b_y})
    if step%1 == 0:
        acc, flat_representation = sess.run([accuracy, flat], {tf_x: b_x, tf_y: b_y})
        print('step', step, '| train loss: %.4f' % loss_, '| accuracy: %.2f' % acc)
