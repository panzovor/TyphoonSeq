import tensorflow as tf
import Dir
import math

class SNW():

    def __init__(self):
        self.data_type = tf.float32
        self.feature_num = 4
        self.hidden_node = 50
        self.num_step =4
        self.lr=  0.01
        self.batch_size= -1

        self.output_node = 2
        self.x = tf.placeholder(tf.float32,shape=[None,self.num_step*self.feature_num])
        self.y = tf.placeholder(tf.float32,shape=[None,self.output_node])
        self.repeat_times =100
        self.inter =3
        self.device ="/cpu:0"


    def network_structure(self):
        w1 = tf.Variable(tf.random_normal([self.num_step*self.feature_num, self.output_node], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([self.output_node], stddev=0.1))
        final_value = tf.matmul(self.x,w1)+b1


        # w2 = tf.Variable(tf.random_normal([self.hidden_node, self.output_node], stddev=0.1))
        # b2 = tf.Variable(tf.random_normal([self.output_node], stddev=0.1))
        # final_value = tf.matmul(hidden_value, w2) + b2

        cost = tf.contrib.losses.mean_squared_error(final_value, self.y)
        optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(cost)


        return optimizer,cost,final_value


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
            tmp = []
            for var in each_data[:-1]:
                tmp.extend(var)
            result[0].append(tmp)
            result[1].append(each_data[-1][0:2])
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
        optimizer,cost,predict = self.network_structure()
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
                    batch_predict = session.run(predict,feed_dict={self.x:batch_data[0],self.y:batch_data[1]})
                    real_loss = self.real_loss(batch_predict,batch_data[1])

                print("repeat time", str(repeat_num), "batch id",str(i),"real_loss",real_loss)


                print(batch_predict,batch_data[1])
                repeat_num+=1
                if testdata != None:
                    test_loss = self.test(testdata,session,predict)
                    print("real test loss", test_loss)

            return batch_predict,batch_data[1]

    def demo(self):
        train_path = Dir.resourceDir + "/smalldata/train.txt"
        test_path = Dir.resourceDir + "/smalldata/test.txt"
        data = self.read_data(train_path)
        testdata = self.read_data(test_path)
        return self.train(data, testdata)
        #
        # repeat_num = 0
        # while repeat_num < self.repeat_times:
        #
        #     step_num_per_round = int(data.__len__() / data.__len__() )
        #     print(step_num_per_round)
        #     for i in range(step_num_per_round):
        #         batch_data = self.next_batch(data, i, self.batch_size)
        #         print(batch_data[0])
        #         print(batch_data[1])
        #     repeat_num+=1

if __name__ == "__main__":
    sls = SNW()
    sls.demo()