import tensorflow as tf
import Dir
import math


class SimpleLstmSeq():


    def __init__(self):
        self.hidden_node = 50
        self.batch_size = 1000
        self.output_node = 2
        self.dtype = tf.float32
        self.num_step = 4
        self.feature_num=4
        self.x = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.y = tf.placeholder(self.dtype,shape=[None,self.output_node])
        self.lr = 0.1
        self.decay = 0.9
        self.repeat_times = 10000
        self.device ="/cpu:0"

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

        with tf.variable_scope("rnn"):
            # tf.get_variable_scope().reuse_variables()
            outputs,states = tf.nn.rnn(lstm_cell,x,dtype=self.dtype)
        output = tf.concat(1,outputs)

        ### full_connected part structure
        w = tf.Variable(tf.random_normal([self.hidden_node*self.num_step,self.output_node],stddev=0.1))
        b = tf.Variable(tf.random_normal([self.output_node],stddev=0.1))
        predict = tf.matmul(output,w)+b

        ### cost
        cost = tf.contrib.losses.mean_squared_error(predict,tf.reshape(self.y,[-1,2]))

        ### optimizer
        optimizer = tf.train.RMSPropOptimizer(self.lr,decay=self.decay).minimize(cost)

        return predict,outputs,cost,optimizer


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
        result = [[],[]]
        for each_data in batch_data:
            result[0].extend(each_data[:-1])
            result[1].append(each_data[-1][0:2])
        return result

    def real_loss(self,predict,y):
        result = []
        # print(predict.__len__())
        # print(y.__len__())
        for i in range(predict.__len__()):
            pre = predict[i]
            y_ = y[i]
            dist = math.sqrt(math.pow(float(pre[0])-float(y_[0]),2) + math.pow(float(pre[1])-float(y_[1]),2))
            result.append(dist)
        return 11*sum(result)/result.__len__()

    def test(self,testdata,session,predict):
        test_data= self.next_batch(testdata)
        real_label = test_data[1].copy()
        test_predict = session.run(predict,feed_dict={self.x:test_data[0],self.y:test_data[1]})
        return self.real_loss(test_predict,real_label)

    def train(self,data,testdata=None):
        predict,hidden_output,cost,optimizer = self.network_struct()
        init = tf.global_variables_initializer()
        if self.batch_size == -1: self.batch_size = data.__len__()

        with tf.device(self.device),tf.Session() as session:
            session.run(init)
            repeat_num =0
            while repeat_num < self.repeat_times:
                step_num_per_round = int(data.__len__()/self.batch_size)
                for i in range(step_num_per_round):
                    batch_data= self.next_batch(data,i,self.batch_size)
                    session.run(optimizer,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                    batch_predict = session.run(predict,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                    real_loss = self.real_loss(batch_predict,batch_data[1])
                print("repeat time", str(repeat_num), "batch id",str(i),"real_loss",real_loss)
                repeat_num+=1
                if testdata != None:
                    test_loss = self.test(testdata,session,predict)
                    print("real test loss", test_loss)

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
                if tmp_data[key].__len__()<self.num_step+1:
                    continue
                else:
                    for i in range(tmp_data[key].__len__()-self.num_step-1):
                        data.append(tmp_data[key][i:i+self.num_step+1])
        return data

if __name__ == "__main__":
    sls = SimpleLstmSeq()
    train_path = Dir.resourceDir+"train.txt"
    test_path = Dir.resourceDir+"test.txt"
    data = sls.read_data(train_path)
    testdata =sls.read_data(test_path)
    tes = sls.next_batch(testdata)
    # for line in tes[1]:
    #     print(line)
    # print(tes[1].__len__())
    sls.train(data,testdata)