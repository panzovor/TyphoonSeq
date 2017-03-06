__author__ = 'E440'
import Dir
import tensorflow as tf
import math

class SeqLstmModel():

    def __init__(self):
        self.num_steps =4
        ### first = 128
        self.batch_size=-1
        self.lstm_nodes = 100
        self.feature_nums = 4
        self.data_type = tf.float32
        self.output_keep_prob = 0.8
        self.lstm_layers = 2
        self.output_nodes = 2
        self.lr = 0.05
        self.device = "/cpu:0"
        self.repeate_train_times = 10
        self.display_per_nums = 100
        self.inter = 3

    ### read data from single file
    ### input : filepath
    ### output: shape = [batch_size, num_steps, feature_nums]
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
                if tmp_data[key].__len__()<self.num_steps+self.inter+1:
                    # count+=1
                    # print(count)
                    continue
                else:
                    for i in range(tmp_data[key].__len__()-self.num_steps-self.inter):
                        real_data = tmp_data[key][i:i+self.num_steps]
                        real_data.append(tmp_data[key][i+self.num_steps+self.inter])
                        data.append(real_data)
        return data

    def seperate_data(self,data,train_rate = 0.8):
        trainsize = int(data.__len__()* train_rate)
        trainData = data[:trainsize]
        testData = data[trainsize:]
        return trainData,testData



    ### definal a multi layer lstm model
    ### inputs: shape = [batch_size, num_steps, feature_nums]
    ### output: shpae = [batch_size*num_steps, lstm_nodes]
    def init_lstm(self):
        x = tf.placeholder(self.data_type,shape=[self.batch_size,self.num_steps,self.feature_nums])
        y = tf.placeholder(self.data_type,shape=[self.batch_size,self.feature_nums-2])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_nodes)
        if self.output_keep_prob>0 and self.output_keep_prob<1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob = self.output_keep_prob)
        if not isinstance(self.lstm_layers,int):
            self.lstm_layers =2
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.lstm_layers,state_is_tuple = True)
        state = cell.zero_state(self.batch_size,self.data_type)

        outputs = []
        with tf.variable_scope("Rnn"):
            for time_step in range(self.num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(x[:,time_step,:],state)
                outputs.append(cell_output)

        final_output = outputs[-1]

        full_connect_w = tf.Variable(tf.random_normal([self.lstm_nodes,self.output_nodes],stddev= 0.1),dtype=self.data_type,name="full_connected_w")
        full_connect_b = tf.Variable(tf.random_normal([self.output_nodes],stddev= 0.05),dtype=self.data_type,name="full_connected_b")

        predict = tf.matmul(final_output,full_connect_w)+full_connect_b
        # predict_ = tf.Variable(predict,name="predict")
        cost = tf.reduce_sum(tf.contrib.losses.mean_squared_error(predict,y))
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train_op = optimizer.minimize(cost)
        # return train_op,cost,x,y,predict_
        return train_op,cost,x,y,predict



    ### definal a multi layer lstm model
    ### inputs: shape = [batch_size, num_steps, feature_nums]
    ### output: shpae = [batch_size*num_steps, lstm_nodes]
    # def init_lstm(self):
    #     x = tf.placeholder(self.data_type,shape=[self.batch_size,self.num_steps,self.feature_nums])
    #     y = tf.placeholder(self.data_type,shape=[self.batch_size,self.feature_nums-2])
    #
    #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_nodes)
    #     output,state = tf.nn.rnn(lstm_cell,x,self.data_type)
    #
    #     final_output = output[-1]
    #
    #     full_connect_w = tf.Variable(tf.random_normal([self.lstm_nodes,self.output_nodes],stddev= 0.1),dtype=self.data_type,name="full_connected_w")
    #     full_connect_b = tf.Variable(tf.random_normal([self.output_nodes],stddev= 0.05),dtype=self.data_type,name="full_connected_b")
    #
    #     predict = tf.matmul(final_output,full_connect_w)+full_connect_b
    #     # predict_ = tf.Variable(predict,name="predict")
    #     cost = tf.reduce_sum(tf.contrib.losses.mean_squared_error(predict,y))
    #     optimizer = tf.train.GradientDescentOptimizer(self.lr)
    #     train_op = optimizer.minimize(cost)
    #     # return train_op,cost,x,y,predict_
    #     return train_op,cost,x,y,predict


    ### input : shape = [batch_size, num_steps, feature_nums]
    ### output:
    def get_next_tarin_data(self,data,train_step):
        index= train_step*self.batch_size
        tmp = data[index:index+self.batch_size]
        # for line in tmp:
        #     print(line)
        x,y = [],[]
        for batch_line in tmp:
            tmp_x = batch_line[:-1]
            tmp_y = batch_line[-1][0:2]
            x.append(tmp_x)
            y.append(tmp_y)
        return x,y

    def cal(self,predict,y):
        result = []
        for i in range(predict.__len__()):
            pre = predict[i]
            y_ = y[i]
            dist = math.sqrt(math.pow(float(pre[0])-float(y_[0]),2) + math.pow(float(pre[1])-float(y_[1]),2))
            result.append(dist)
        return 11*sum(result)/result.__len__()

    def save(self,session,name,repeat_num ):
        filepath = Dir.resourceDir + "model/model_batch_szie_"+str(self.batch_size)+"_mid_node_"+self.lstm_nodes+"_repeat_num_"+str(repeat_num)+"_loss_"+name+".ckpt"
        saver = tf.train.Saver()
        saver.save(session, filepath)

    def load(self,filepath):
        train_op, cost, x, y, predict_ = self.init_lstm()
        saver = tf.train.Saver()
        session = tf.Session()
        saver.restore(session, filepath)
        return train_op,cost,x,y,predict_,session

    def train(self,trainData,testData):
        if self.batch_size == -1:
            self.batch_size = trainData.__len__()
        train_op,cost,x,y,predict_ = self.init_lstm()
        init= tf.global_variables_initializer()
        # trainData,testData = self.seperate_data(data)

        train_loss =10000
        test_loss = 5000
        count =1
        with tf.device(self.device),tf.Session() as session:
            session.run(init)
            repeat_num =0
            train_num_per_round = int(trainData.__len__()/self.batch_size)
            test_num_per_round = int(testData.__len__()/self.batch_size)
            # print(train_num_per_round,test_num_per_round)
            predicts = None
            y_ = None
            while repeat_num <= 350:
                train_step =0
                train_loss = [[],[]]
                while train_step< train_num_per_round:
                    x_,y_ = self.get_next_tarin_data(trainData,train_step)
                    session.run(train_op,feed_dict = {x: x_,y: y_})
                    pred = session.run(predict_, feed_dict={x: x_, y: y_})
                    train_loss[0].extend(y_)
                    train_loss[1].extend(pred)
                    train_step+=1
                train_loss = self.cal(train_loss[0],train_loss[1])

                predicts = session.run(predict_,feed_dict = {x: x_,y: y_})
                # print(y_[0])
                #
                # print(predicts[0])
                # print(predicts[-1])
                # print(y_[-1])
                # input()

                print("train loss", train_loss)
                test_step = 0
                test_loss=[[],[]]
                while test_step< test_num_per_round:
                    x_, y_ = self.get_next_tarin_data(testData, test_step)
                    pred = session.run(predict_, feed_dict={x: x_, y: y_})
                    test_loss[0].extend(y_)
                    test_loss[1].extend(pred)
                    test_step+=1
                test_loss = self.cal(test_loss[0], test_loss[1])
                print(str(repeat_num)+"test loss", test_loss)
                repeat_num+=1
            # self.save(session,test_loss,repeat_num)
            return predicts,y_



    def test(self,data,filepath="/home/czb/PycharmProjects/TyphoonSeq/data/model/model.ckpt"):
        train_op, cost, x, y, predict_,session = self.load(filepath)
        test_step = 0
        test_loss = [[], []]
        # data = self.read_data()
        # trainData, testData = self.seperate_data(data)
        test_num_per_round = int(data.__len__() / self.batch_size)
        while test_step < test_num_per_round:
            x_, y_ = self.get_next_tarin_data(data, test_step)
            pred = session.run(predict_, feed_dict={x: x_, y: y_})
            test_loss[0].extend(y_)
            test_loss[1].extend(pred)
            test_step += 1
        test_loss = self.cal(test_loss[0], test_loss[1])
        print("test loss", test_loss)

    def demo(self):
        train_data = self.read_data(filepath="/home/czb/PycharmProjects/TyphoonSeq/data/smalldata/train.txt")
        test_data =  self.read_data(filepath="/home/czb/PycharmProjects/TyphoonSeq/data/smalldata/test.txt")
        return self.train(train_data, test_data)

if __name__ == "__main__":
    model = SeqLstmModel()
    train_data= model.read_data(filepath="/home/czb/PycharmProjects/TyphoonSeq/data/smalldata/train.txt")
    test_data = model.read_data(filepath="/home/czb/PycharmProjects/TyphoonSeq/data/smalldata/test.txt")
    test_data = model.train(train_data,test_data)
    # model.test(test_data)
    # print(data.__len__())