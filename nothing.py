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

# def test():
#     x =0
# 
# 
# class a():
#     def __init__(:
#         x =0
# 
#     def change(d):
# A  =a()
# A.x
def ha(each_data):
    window_size = 1
    window_num = 10

    previous_position = each_data[-2]
    after_position = each_data[-1]
    window_index = window_num+int((after_position[0]- previous_position[0]))/window_size

    row_index = window_num-int((after_position[1]-previous_position[1]))/window_size

    location_index = window_index+window_num*row_index*2
    print(location_index)

    x_predict = int(location_index/(window_num*2))
    y_predict = int(location_index%(window_num*2))
    print(x_predict,y_predict)

    return location_index
# result[1].append([location_index])
#
# data= [[0,0],[-10,10]]
# data1= [[0,0],[9,10]]
# data2= [[0,0],[-10,-9]]
# data3= [[0,0],[9.5,-9]]
data4= [[0,0],[0,0]]
# ha(data)
# ha(data1)
# ha(data2)
# ha(data3)
ha(data4)