import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow

x_data = pd.read_csv('MNIST_15_15.csv', header=None)
x_data = np.array(x_data,dtype=float)
y_data = pd.read_csv('MNIST_LABEL.csv', header=None)
y_data = np.array(y_data,dtype=float)
y_data = y_data.flatten()-5

print(y_data.shape)
print(x_data.shape)

xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.int32, y_data.shape)


#add layers
l1 = tf.layers.dense(xs, 1024, tf.nn.relu)
output = tf.layers.dense(l1, 2)


#the errow between prediction and real data
loss = tf.losses.sparse_softmax_cross_entropy(labels=ys, logits=output)
tf.summary.scalar('loss',loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.squeeze(ys), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)


sess = tf.Session()                                                                 # control training and others
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/',sess.graph)
sess.run(init_op)     # initialize var in graph


for step in range(100):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output], {xs: x_data, ys: y_data})
    if step % 2 == 0:
        # plot and show learning process
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result,step)
        print(acc)
'''
for i in range(5):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
    writer.add_summary(result,i)
    if i ==0:
        pass
#p = (sess.run(prediction,feed_dict={xs:x_data}))
#print(p)
'''
