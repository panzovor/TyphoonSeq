__author__ = 'E440'
import tensorflow as tf
import Dir
import math


class SimpleLstmSeqClass():


    def __init__(self):


        ### 前后各一百个格子
        self.window_num =20
        ### 每个格子大小为1经纬度，约110公里
        self.window_size=10
        self.hidden_node = 50
        self.batch_size = 128
        self.output_node = 2
        self.dtype = tf.float32
        self.num_step = 4
        self.feature_num=4
        self.x = tf.placeholder(self.dtype,shape=[None,self.feature_num])
        self.y = tf.placeholder(self.dtype,shape=[None,self.window_num*self.window_num*4])
        self.lr = 0.01
        self.decay = 0.9
        self.repeat_times = 1000
        self.device ="/cpu:0"
        self.inter = 1

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
        w = tf.Variable(tf.random_normal([self.hidden_node*self.num_step,self.window_num*self.window_num*4],stddev=0.1))
        b = tf.Variable(tf.random_normal([self.window_num*self.window_num*4],stddev=0.1))
        predict = tf.matmul(output,w)+b
        predict = tf.sigmoid(predict)
        ### cost
        cost = tf.contrib.losses.mean_squared_error(predict,self.y)
        # cost = tf.nn.sigmoid_cross_entropy_with_logits(predict,self.y)


        ### optimizer
        optimizer = tf.train.RMSPropOptimizer(self.lr,decay=self.decay).minimize(cost)
        return predict,outputs,cost,optimizer,states

    def map2index(self,previous_position,after_position):
        window_index = int(self.window_num+int((after_position[0]- previous_position[0]))/self.window_size)
        row_index = int(self.window_num-int((after_position[1]-previous_position[1]))/self.window_size)
        location_index = int(window_index+self.window_num*row_index*2)
        result= [0.0 for var in range(self.window_num*self.window_num*4)]
        if location_index <0 or location_index >= result.__len__():
            print(location_index,result.__len__(),previous_position,after_position,window_index,row_index)
            input()
        result[location_index] =1.0

        return result

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
                    tmp_data[key_].append([float(var) for var in tmp[:]])
            for key in tmp_data.keys():
                # if tmp_data[key].__len__()<self.num_step:
                if tmp_data[key].__len__()<self.num_step+self.inter+1:
                    continue
                else:
                    for i in range(tmp_data[key].__len__() - self.num_step-self.inter):
                        real_data = tmp_data[key][i:i+self.num_step]
                        real_data.append(tmp_data[key][i+self.num_step+self.inter])
                        # print(i,i+self.num_step,i+self.num_step+self.inter)
                        data.append(real_data)
        return data

    def preprocess_each_data(self,each_data):
        result =[]
        if isinstance(each_data[0],list):
            for each in each_data:
                result.append(each[-4:])
        else:
            result = each_data[-4:]
        return result


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
        result = [[],[],[],[]]
        for each_data in batch_data:
            result[0].extend(self.preprocess_each_data(each_data[:-1]))
            previous_position = self.preprocess_each_data(each_data[-2])
            after_position = self.preprocess_each_data(each_data[-1])
            location_index = self.map2index(previous_position,after_position)
            result[2].append(after_position[0:2])
            result[3].append(previous_position[0:2])
            result[1].append(location_index)
        return result

    # def get_all_data(self,data):

    def calculate_killo(self,predict_index,previous_position,real_position):
        x_predict = int(predict_index/(self.window_num*2))
        y_predict = int(predict_index%(self.window_num*2))

        x_previous = self.window_num
        y_previous = self.window_num

        predict_longtitude = self.window_size*(x_predict - x_previous)+previous_position[0]-5
        predict_latitude = self.window_size*(y_predict - y_previous)+previous_position[1]-5

        dist_x = abs(predict_longtitude - real_position[0])
        dist_y = abs(predict_latitude - real_position[1])

        dist = math.sqrt(math.pow(dist_x,2) + math.pow(dist_y,2))
        return dist

    def real_loss(self,predict,previous,y):
        result = []

        for i in range(predict.__len__()):
            predict_list= list(predict[i])
            predict_ = predict_list.index(max(predict_list))
            previous_ = [previous[i][0] , previous[i][1]]
            real_ = [y[i][0],y[i][1]]
            dist = self.calculate_killo(predict_,previous_,real_)
            result.append(dist)
        return sum(result)/result.__len__()

    def test(self,testdata,session,predict):
        test_data= self.next_batch(testdata)
        test_predict = session.run(predict,feed_dict={self.x:test_data[0]})
        return self.real_loss(test_predict,test_data[3],test_data[2])

    def train(self,data,testdata=None):
        predict,hidden_output,cost,optimizer,states = self.network_struct()
        init = tf.global_variables_initializer()
        if self.batch_size == -1: self.batch_size = data.__len__()
        min_loss = 109.6521987921885
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
                    batch_predict = session.run(predict,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                    real_loss = self.real_loss(batch_predict,batch_data[3],batch_data[2])
                    train_cost= session.run(cost,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                # print("repeat time", str(repeat_num),train_cost)
                    print("repeat time", str(repeat_num),"batch id",i, "train cost",train_cost,"batch len",step_num_per_round, "real_loss",real_loss)


                repeat_num+=1
                # data_train_all  = self.next_batch(data,0,batch_size=-1)
                # batch_predict = session.run(predict, feed_dict={self.x: data_train_all[0], self.y: data_train_all[1]})
                # real_loss = self.real_loss(batch_predict, data_train_all[3], data_train_all[2])
                # print("repeat time", str(repeat_num), "real_loss", real_loss)
                if testdata != None:
                    test_loss = self.test(testdata,session,predict)
                    print("real test loss", test_loss)

                    if test_loss < min_loss:
                        print("saveing hidden value")
                        data_train_all1 = self.next_batch(data, 0, batch_size=-1)
                        hidden_train = session.run(hidden_output,feed_dict={self.x:data_train_all1[0],self.y:data_train_all1[1]})
                        self.save(data,hidden_train,path = Dir.resourceDir+"hidden/train.txt")

                        data_test_all1 = self.next_batch(testdata, 0, batch_size=-1)
                        hidden_test = session.run(hidden_output, feed_dict={self.x: data_test_all1[0], self.y: data_test_all1[1]})
                        self.save(testdata,hidden_test, path=Dir.resourceDir + "hidden/test.txt")
                        min_loss = test_loss

            return batch_predict,batch_data[1]


    def demo(self):
        train_path = Dir.resourceDir + "/smalldata/train.txt"
        test_path = Dir.resourceDir + "/smalldata/test.txt"
        data = self.read_data(train_path)
        testdata = self.read_data(test_path)
        return self.train(data, testdata)

    # def save(self,hidden_value):

    def save(self,data,hidden,path):
        line = ""
        count =0
        print(data.__len__(),hidden[0].__len__(),self.num_step+self.inter)
        with open(path,mode="w",encoding="utf-8") as file:
            for i in range(hidden[0].__len__()):
                line += str(data[count][0])[1:-1]+","+str(list(hidden[0][i]))[1:-1]+"\n"
                count+=1
                # tmp.append( str(list(hidden[0][i])))
            for i in range(self.num_step+self.inter):
                line += str(data[-1][i])[1:-1]+","+str(list(hidden[-1][-1-i]))[1:-1]+"\n"
                count+=1
                # tmp.append(str(list(hidden[0][i])))
            file.write(line)
            file.flush()
            print(count)



if __name__ == "__main__":
    sls = SimpleLstmSeqClass()
    train_path = Dir.resourceDir+"alldata/train.txt"
    test_path = Dir.resourceDir+"alldata/test.txt"
    data = sls.read_data(train_path)
    testdata =sls.read_data(test_path)
    # tes = sls.next_batch(testdata)
    # for line in tes[1]:
    #     print(line)
    # print(tes[1].__len__())
    sls.train(data,testdata)