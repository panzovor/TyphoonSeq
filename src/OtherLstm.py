import tensorflow as tf
import Dir
import math


class SimpleLstmSeq():


    def __init__(self):
        self.hidden_node = 50
        self.batch_size = -1
        self.output_node = 2
        self.dtype = tf.float32
        self.num_step = 4
        self.feature_num=4
        self.x = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.y = tf.placeholder(self.dtype,shape=[None,self.output_node])
        self.lr = 0.05
        self.decay = 0.9
        self.repeat_times = 1000
        self.device ="/cpu:0"
        self.inter = 3
        # self.label_index = 2
        self.spacename = "rnn"

    def network_struct(self):
        ### preprocess the place holder data
        ### origin shape =[batch size, feature num]
        ### final  shape =[num_step, batch size, feature num]
        x = tf.reshape(self.x,shape =[self.num_step,-1,self.feature_num])
        x = tf.transpose(x,[1,0,2])
        x = tf.reshape(x,[-1,self.feature_num])
        x = tf.split(0,self.num_step,x)

        ### lstm part structure
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_node)

        lstm_cells =tf.nn.rnn_cell.MultiRNNCell([lstm_cell for i in range(self.num_step)], state_is_tuple=True)
        outputs,states = tf.nn.rnn(lstm_cells,x,dtype=self.dtype)
        # predict,loss = tf.contrib.learn.models.linear_regression(outputs[-1],self.y)
        outputs = tf.concat(1, outputs)
        w = tf.Variable(tf.random_normal([self.hidden_node*self.num_step,self.output_node],stddev=0.1))
        b = tf.Variable(tf.random_normal([self.output_node],stddev=0.1))
        predict = tf.matmul(outputs,w)+b

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,self.y))

        train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer= "Adagrad",learning_rate = self.lr)


        return predict,outputs,loss,train_op


    ### data shape = [data num, num step, feature num]
    ### output =  x, y
    ### x shape = [batch size, 0:-1, feature_num]
    ### y shaep = [batch size, -1 ,  0:2]
    def next_batch(self,data,step=0,batch_size = -1):
        if batch_size == -1:
            batch_size = data.__len__()
        start_index = step* batch_size
        end_index= start_index+batch_size
        if end_index > data.__len__():
            end_index = data.__len__()
        batch_data= data[start_index:end_index]
        result = [[],[],[]]
        for each_data in batch_data:
            result[0].extend(each_data[:-1])
            result[1].append(each_data[-1][0:2])
            # result[2].append([each_data[-1][1]])

        return result

    def real_loss(self,predict,y):
        result = []

        for i in range(predict.__len__()):
            pre = [predict[i][0] , predict[i][1]]
            y_ = y[i]
            dist = math.sqrt(math.pow(float(pre[0])-float(y_[0]),2) + math.pow(float(pre[1])-float(y_[1]),2))
            result.append(dist)
        return 11*sum(result)/result.__len__()

    def test(self,testdata,session,predict):
        test_data= self.next_batch(testdata)
        real_label = test_data[1].copy()
        test_predict = session.run(predict,feed_dict={self.x:test_data[0]})


        return self.real_loss(test_predict,real_label)


    def train(self,data,testdata=None):
        predict, hidden_output, cost, optimizer = self.network_struct()
        init = tf.global_variables_initializer()
        if self.batch_size == -1: self.batch_size = data.__len__()
        with tf.device(self.device),tf.Session() as session:
            session.run(init)
            repeat_num =0
            while repeat_num < self.repeat_times:
                step_num_per_round = int(data.__len__()/self.batch_size)
                for i in range(step_num_per_round):
                    batch_data= self.next_batch(data,i,self.batch_size)
                    y_ = batch_data[1]
                    session.run(optimizer,feed_dict={self.x:batch_data[0],self.y:y_})
                    # session.run(optimizer1,feed_dict={self.x:batch_data[0],self.y:batch_data[2]})
                    batch_predict = session.run(predict,feed_dict={self.x:batch_data[0],self.y:y_})
                    outputs = session.run(hidden_output,feed_dict={self.x:batch_data[0],self.y:y_})
                    # batch_predict1 = session.run(predict,feed_dict={self.x:batch_data[0],self.y:batch_data[2]})
                    real_loss = self.real_loss(batch_predict,batch_data[1])
                    # real_loss = self.real_loss(batch_predict,batch_predict1,batch_data[1],batch_data[2])
                    tmp  = outputs[-1]
                    model_loss= session.run(cost,feed_dict={self.x:batch_data[0],self.y:y_})
                    print(batch_predict,y_)
                    print("----model loss----",model_loss)
                print("repeat time", str(repeat_num), "batch id",str(i),"real_loss",real_loss)

                repeat_num+=1
                # if testdata != None:
                #     test_loss = self.test(testdata,session,predict,predict1,cost)
                #     print("real test loss", test_loss)
                # if testdata != None:
                #     test_loss = self.test(testdata,session,predict)
                #     print("real test loss", test_loss)
            return batch_predict,y_

    def read_data(self,filepath = Dir.resourceDir+"typhoon_route_list_wind_fix.txt"):
        data = []
        tmp_data = {}
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
                # if tmp_data[key].__len__()<self.num_step:
                if tmp_data[key].__len__()<self.num_step+self.inter+1:
                    continue
                else:
                    for i in range(tmp_data[key].__len__() - self.num_step-self.inter):
                        real_data = tmp_data[key][i:i+self.num_step]
                        real_data.append(tmp_data[key][i+self.num_step+self.inter])
                        data.append(real_data)
        return data

    def demo(self):

        train_path = Dir.resourceDir + "/smalldata/train.txt"
        test_path = Dir.resourceDir + "/smalldata/test.txt"
        data = self.read_data(train_path)
        testdata =self.read_data(test_path)
        tes = self.next_batch(testdata)
        # for line in tes[1]:
        #     print(line)
        # print(tes[1].__len__())
        return self.train(data, testdata)

if __name__ == "__main__":
    sls = SimpleLstmSeq()
    sls.demo()