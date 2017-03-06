import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import math
import Dir
class test1():

    def __init__(self):
        self.device = "/cpu:0"
        self.n_step = 4
        self.n_hidden =50
        self.dtype = tf.float32
        self.x = tf.placeholder(self.dtype, [None, self.n_step,4])
        self.y = tf.placeholder(self.dtype, [None,2])
        self.n_output =2
        self.batch_size =100
        self.repeat_num =10000

    def read_data(self,filepath = Dir.resourceDir+"typhoon_route_list_wind_fix.txt"):
        data = []
        tmp_data = {}
        count =0
        with open(filepath,mode="r",encoding="utf-8") as file:
            for line in file.readlines():
                line = line.strip()
                tmp = line.split(",")
                if tmp.__len__() == 7:
                    key_ = tmp[0]
                    if key_ not in tmp_data.keys():
                        tmp_data[key_] = []
                    tmp_data[key_].append(tmp[-4:])
            for key in tmp_data.keys():
                # print(tmp_data[key].__len__())
                if tmp_data[key].__len__()<self.n_step+1:
                    # count+=1
                    # print(count)
                    continue
                else:
                    for i in range(tmp_data[key].__len__()-self.n_step-1):
                        data.append(tmp_data[key][i:i+self.n_step+1])
        return data

    ### x  =shape =[batch,n_step,n_input]
    def rnn_lstm(self,X, W, B,t=0):
        X = tf.split(1, self.n_step, X)
        with tf.variable_scope("mlstm"):
            if t>0:
                tf.get_variable_scope().reuse_variables()
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
            outputs, states = tf.nn.rnn(lstm_cell, X, dtype=tf.float32)
            output = outputs[-1]
        preds = tf.matmul(output, W) + B

        return preds, outputs


    def initialize(self):
        weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_output], stddev=0.1))
        biases = tf.Variable(tf.random_normal([self.n_output], stddev=0.05))
        preds, states = self.rnn_lstm(self.x, weights, biases)
        cost = tf.contrib.losses.mean_squared_error(preds, self.y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9).minimize(cost)
        return optimizer,cost,states,preds

    def next_batch(self,data,step):
        index =self.batch_size*step
        if data.__len__() <index+self.batch_size and step!=0:
            index = data.__len__()-self.batch_size
        x,y=[],[]
        for line in data[index:index+self.batch_size]:
            x.append(line[:-1])
            y.append(line[-1][0:2])
        return x,y

    def cal(self,predict,y):
        result = []
        for i in range(predict.__len__()):
            pre = predict[i]
            y_ = y[i]
            dist = math.sqrt(math.pow(float(pre[0])-float(y_[0]),2) + math.pow(float(pre[1])-float(y_[1]),2))
            result.append(dist)
        return 11*sum(result)/result.__len__()

    def train(self,data):
        self.batch_size = data.__len__()
        optimizer, cost, states,preds = self.initialize()
        init = tf.initialize_all_variables()
        result = [[],[]]
        with tf.device(self.device), tf.Session() as session:
            session.run(init)
            epoch = 1
            training_iter = len(data) / self.batch_size + 1
            while epoch <= self.repeat_num:
                step = 1
                while step < training_iter:
                    batch_x, batch_y = self.next_batch(data, step)
                    # for data in batch_x:
                    #     print(data)
                    session.run(optimizer,feed_dict={self.x: batch_x, self.y: batch_y})
                    predict = session.run(preds,feed_dict={self.x: batch_x, self.y: batch_y})
                    result[0].extend(predict)
                    result[1].extend(batch_y)
                    step += 1
                epoch += 1
                loss =self.cal(result[0],result[1])
                print("repeat times:"+str(epoch)+" loss: "+str(loss))


if __name__ =="__main__":
    mode = test1()
    train_data= mode.read_data(filepath=Dir.resourceDir+"train.txt")
    mode.train(train_data)