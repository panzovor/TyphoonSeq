import tensorflow as tf
import math
import Dir
import src.SimpleLstm_bak as sb

class EDcoder(sb.SimpleLstmSeq):

    def __init__(self):
        self.hidden_node = 50
        self.feature_num=4
        self.batch_size = -1
        self.dtype = tf.float32
        self.x = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.y = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.x_decoder = tf.placeholder(self.dtype,shape=[None,self.hidden_node])
        self.y_decoder = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.inter = 0
        self.repeat_times = 5000
        self.num_step = 4
        self.lr = 0.01
        self.decay = 0.9
        self.repeat_times = 1000
        self.device = "/cpu:0"


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
                        # real_data.append(tmp_data[key][i+self.num_step+self.inter])
                        data.append(real_data)
        return data


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
            result[0].extend(each_data)
            tmp =[]
            tmp.append(each_data)
            if tmp.__len__() == self.num_step:
                result[1].append(tmp.copy())
                tmp =[]

        return result

    def network_struct(self):
        ### encoder
        x = tf.reshape(self.x,shape =[self.num_step,-1,self.feature_num])
        x = tf.transpose(x,[1,0,2])
        x = tf.reshape(x,[-1,self.feature_num])
        x = tf.split(0,self.num_step,x)
        y = tf.reshape(self.y,shape=[self.num_step,-1,self.feature_num])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_node)
        hidden_value = None
        with tf.variable_scope("rnn_encoder"):
            # tf.get_variable_scope().reuse_variables()
            outputs,states = tf.nn.rnn(lstm_cell,x,dtype=self.dtype)
            hidden_value=outputs

        ### decoder
        x_decoder = hidden_value
        lstm_cell_decoder = tf.nn.rnn_cell.BasicLSTMCell(self.feature_num)
        with tf.variable_scope("rnn_decoder"):
            # tf.get_variable_scope().reuse_variables()
            outputs_decoder,states_decoder = tf.nn.rnn(lstm_cell_decoder,x_decoder,dtype=self.dtype)
        outputs_decoder = tf.pack(outputs_decoder)
        cost = tf.contrib.losses.mean_squared_error(outputs_decoder,y)
        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=self.decay).minimize(cost)
        return optimizer,cost,hidden_value

    def train(self,data,testdata=None):
        optimizer, cost, hidden_value = self.network_struct()
        init = tf.global_variables_initializer()
        if self.batch_size == -1: self.batch_size = data.__len__()

        with tf.device(self.device),tf.Session() as session:
            session.run(init)
            repeat_num =0
            batch_predict = None
            batch_data = None
            while repeat_num < self.repeat_times:
                step_num_per_round = int(data.__len__()/self.batch_size)
                for i in range(step_num_per_round):
                    batch_data= self.next_batch(data,i,self.batch_size)
                    session.run(optimizer,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                    real_loss = session.run(cost,feed_dict={self.x:batch_data,self.y:batch_data})
                print("repeat time", str(repeat_num), "batch id",str(i),"real_loss",real_loss)
                repeat_num+=1

            return batch_predict,batch_data

    def demo(self):
        train_path = Dir.resourceDir + "/smalldata/train.txt"
        test_path = Dir.resourceDir + "/smalldata/test.txt"
        data = self.read_data(train_path)
        testdata = self.read_data(test_path)
        return self.train(data, testdata)

if __name__ == "__main__":
    sls = EDcoder()
    train_path = Dir.resourceDir+"/smalldata/train.txt"
    test_path = Dir.resourceDir+"/smalldata/test.txt"
    data = sls.read_data(train_path)
    testdata =sls.read_data(test_path)

    batch_data= sls.next_batch(data)
    print(batch_data[0])
    print(batch_data[1])

    # sls.train(data)
    # sls.demo()