import tensorflow as tf
a = 1
for i in range(1000000):
    a+=0.000001
print (a-1)