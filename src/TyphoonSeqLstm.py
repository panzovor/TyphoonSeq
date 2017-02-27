__author__ = 'E440'
import Dir
import tensorflow as tf

class SeqLstmModel():

    def __init__(self):
        self.num_steps =4
        self.batch_size= 1
        self.lstm_nodes = 50
        self.feature_nums = 4
        self.data_type = tf.float32
        self.output_keep_prob = 0.8
        self.lstm_layers = 2
        self.output_nodes = 2
        self.lr = 0.05
        self.device = "/cpu:0"
        self.repeate_train_times = 10
        self.display_per_nums = 100

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
                if tmp_data[key].__len__()<self.num_steps+1:
                    # count+=1
                    # print(count)
                    continue
                else:
                    for i in range(tmp_data[key].__len__()-self.num_steps-1):
                        data.append(tmp_data[key][i:i+self.num_steps+1])
        return data

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

    def train(self):
        train_op,cost,x,y,predict_ = self.init_lstm()
        init= tf.global_variables_initializer()
        data= self.read_data()
        with tf.device(self.device),tf.Session() as session:
            session.run(init)
            repeat_num =0
            train_num_per_round = data.__len__()/self.batch_size
            while repeat_num<self.repeate_train_times:
                train_step =0
                while train_step< train_num_per_round:
                    x_,y_ = self.get_next_tarin_data(data,train_step)
                    session.run(train_op,feed_dict = {x: x_,y: y_})
                    # print(x_,y_)
                    if train_step%self.display_per_nums == 0:
                        loss = session.run(cost,feed_dict={x:x_,y:y_})
                        print("repeat num"+str(repeat_num)+" train_times: "+str(int(train_step/self.display_per_nums))
                        +" loss = "+str(loss*22))
                    pred = session.run(predict_,feed_dict={x:x_,y:y_})
                    # print(pred,y_)
                    # print()
                    train_step+=1
                repeat_num+=1


if __name__ == "__main__":
    model = SeqLstmModel()
    data= model.read_data()

    # x,y = model.get_next_tarin_data(data,0)
    # print(x)
    # print(y)
    # for line in data:
    #     print(line)
    model.train()
    print(data.__len__())