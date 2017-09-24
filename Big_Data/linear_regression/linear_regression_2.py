import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mnist = pd.read_csv("MNIST_15_15", header = None)
mnist_label = pd.read_csv("MNIST_LABEL", header = None)

data_set = []
print (mnist)