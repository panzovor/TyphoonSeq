import math

x=3
y=4
print(math.sqrt(math.pow(x,2)+math.pow(y,2)))

import tensorflow as tf

x= [1,2,3,4,5,6,7,8,9]
y = [11,22,33,44,55,66,77,88,99]
result = tf.reshape(tf.concat(0,[x,y]),shape=[-1,2])
tf.Print(result,[result])
#
# import numpy as np
#
# result = np.sin(np.linspace(0,100,1000))
# print(result)

def test():
    x =0


class a():
    def __init__(self):
        self.x =0

    def change(self,d):
A  =a()
A.x