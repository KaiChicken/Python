import random
import sys
import os
import numpy as np
quote = "\"Always remember you are unique"

multi_line_quote = '''just 
like everyone else'''

print("%s %s %s" % ('I like the quote', quote, multi_line_quote))

print("I don't like ", end="")
print("newlines " * 5)

a = np.array([[1,2],[3,4]])
std = np.std(a)
print (np.mean(a))