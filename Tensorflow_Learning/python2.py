import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)

### method1
sess = tf.Session()
result = sess.run(product)  #<---- you must have sess.run() everytime you want to use anything of tensorflow
print(result)
sess.close()

#method 2
#sess would be automatically closed after it run
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
