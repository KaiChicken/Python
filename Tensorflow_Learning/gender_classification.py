"""
Deopendencies:
tensorflow: 1.1.0
matplotlib
numpy
pandas
python gender_classification.py
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("wiki5.csv", index_col=0)
x_data = data.drop(['gender', 'age'], axis=1)
y_data = data['gender']
x_data = np.array(x_data,dtype=float)/255
y_data = np.array(y_data,dtype=float)

tf_x = tf.placeholder(tf.float32, x_data.shape)
tf_y = tf.placeholder(tf.int32, y_data.shape)
tf_is_training = tf.placeholder(tf.bool, None)

#neural network layers
l1 = tf.layers.dense(tf_x, x_data.shape[1], tf.nn.relu)
l1 = tf.layers.dropout(l1, rate=0.5, training=tf_is_training)
l2 = tf.layers.dense(l1, x_data.shape[1], tf.nn.relu)
l2 = tf.layers.dropout(l2, rate=0.5, training=tf_is_training)
output = tf.layers.dense(l2, 2)

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy = tf.metrics.accuracy(
    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)
tf.summary.scalar('loss',loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

writer = tf.summary.FileWriter('./log', sess.graph)
merge_op = tf.summary.merge_all()

#train data
for step in range(100):
    _, acc, pred, result = sess.run([train_op, accuracy, output, merge_op], {tf_x: x_data, tf_y:y_data, tf_is_training:True})
    writer.add_summary(result, step)
    if step % 10 == 0:
        print(acc)
