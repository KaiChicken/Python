import random
import sys
import os
import numpy as np
import pandas as pd
quote = "\"Always remember you are unique"

multi_line_quote = '''just 
like everyone else'''

print("%s %s %s" % ('I like the quote', quote, multi_line_quote))

print("I don't like ", end="")
print("newlines " * 5)

a = np.array([[1,2],[3,4]])
std = np.std(a)
print (np.mean(a))

a = np.arange(20)
print(a)

a = [[1,2,3],[2,3,4],[3,4,5]]
a = pd.DataFrame(a)

import tkinter as tk
window = tk.Tk()
window.title('my window')
window.geometry('200x100')

l = tk.Label(window,
             text='OMG! this is TK!',    # 标签的文字
             bg='green',     # 背景颜色
             font=('Arial', 12),     # 字体和字体大小
             width=15, height=2  # 标签长宽
             )
l.pack()    # 固定窗口位置
window.canvas
window.mainloop()