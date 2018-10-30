'''创建服务器端程序，用来接收客户端传进的数据'''
 
from socket import *
import numpy as np
from network_elements import *
import json
from jsocket import Jsocket


class Sever:
    def __init__(self, HOST, PORT, listen_num):
        self.HOST = HOST
        self.PORT = PORT
        self.listen_num = listen_num

    def start(self):
        ADDR = (self.HOST, self.PORT)
        server_socket = socket(AF_INET, SOCK_STREAM)
        server_socket.bind(ADDR)
        server_socket.listen(self.listen_num)
        while True:
            print('Waiting for connecting ......')
            tcpclientsocket, addr = server_socket.accept()
            print('Connected by ', addr)

            ##one process
            while True:
                ## get net config
                network = []
                while True:
                    ## initialize network first
                    net_config = self.get_net_config(tcpclientsocket)
                    self.start_parameter(net_config)

                    learning_rate = 0.1
                    try:
                        network = Network(net_config, learning_rate)
                    except IndexError as e:
                        continue
                    except Exception as e:
                        continue
                    break


                ## get train file
                [x_train, y_train, x_test, y_test, data_range] = self.get_all_data(tcpclientsocket)

                ## get a container
                container = Container(network)

                ## begin training
                try:
                    tcpclientsocket.send("Training...".encode())
                    container.train(x_train, y_train, 50)
                except ValueError as e:
                    print("Training failed, please try all again!")
                    tcpclientsocket.send("You have put in wrong net config, please tr"
                                         "y again! \nType in try again!\n".encode())
                    continue
                except Exception as e:
                    print("Training failed, please try all again!")
                    tcpclientsocket.send("You have put in wrong net config, please tr"
                                         "y again! \nType in try again!\n".encode())
                    continue

                tcpclientsocket.send("Testing...".encode())
                loss = container.test(x_test, y_test)
                print("Test loss: " + str(loss))
                tcpclientsocket.send(("Test loss: " + str(loss) + "\n").encode())
                tcpclientsocket.send("type in gre,gpa,rank: ".encode())

                ## predict
                while True:
                    data = tcpclientsocket.recv(1024)
                    if not data:
                        continue
                    if data.decode() == "exit":
                        break
                    x_input = (data.decode()).split(",")
                    print("predict input: " , x_input)
                    x_input = [float(num) for num in x_input]
                    x_input = [x_input[i] / data_range[i] for i in range(len(x_input))]
                    y_predit = container.predict(x_input)
                    tcpclientsocket.send(("prediction for admission: "
                                          "" + str(y_predit[0,1]) + "\ntype in gre,gpa,rank: ").encode())
                    print("prediction for admission: " + str(y_predit[0,1]))
                break
            tcpclientsocket.close()
        server_socket.close()

    @staticmethod
    def read_file(file_path):
        ## file_path = "..///data//student_data.csv"
        file = open(file_path, encoding='utf-8-sig')
        first_line_flag = True
        tags = []
        datas = []
        for line in file:
            if first_line_flag:
                first_line_flag = False
                continue
            line = line.strip()
            parts = line.split(",")
            tag = int(parts[0])
            tags.append(tag)
            data = [float(parts[1]), float(parts[2]), float(parts[3])]
            datas.append(data)

        file.close()
        tags = np.array(tags) * 1.0
        tags = tags.reshape(tags.shape[0], 1)
        tags_final = np.zeros([tags.shape[0], 2])
        for i in range(tags.shape[0]):
            j = int(tags[i][0])
            tags_final[i][j] = 1.0
        tags = tags_final
        datas = np.array(datas) * 1.0

        ## 做归一化处理
        data_range = []
        for i in range(datas.shape[1]):
            column_max = np.max(datas[:, i])
            column_min = np.min(datas[:, i])
            data_range.append(column_max-column_min)
            datas[:, i] = datas[:, i] * 1.0 / (column_max - column_min)

        np.save("..//data//tags.npy", tags)
        np.save("..//data//datas.npy", datas)

        tags = np.load("..//data//tags.npy")
        datas = np.load("..//data//datas.npy")
        m = tags.shape[0]
        train_idx = int(m * 0.8)
        x_train = datas[:train_idx, :]
        y_train = tags[:train_idx, :]
        x_test = datas[train_idx:, :]
        y_test = tags[train_idx:, :]

        return [x_train, y_train, x_test, y_test,data_range]

    @staticmethod
    def get_net_config(tcpclientsocket):
        while True:
            tcpclientsocket.send("type in the net config for network (for example 3,30,1,2,-1): ".encode())
            data = tcpclientsocket.recv(1024)
            if not data:
                continue
            data = data.decode()
            data = data.strip("[]")
            data = data.split(",")
            try:
                net_config = [int(i) for i in data]
            except ValueError:
                continue
            break
        return net_config

    def get_all_data(self, tcpclientsocket):
        while True:
            tcpclientsocket.send("type in the file path for training this network "
                                 "(for example: ..//data//student_data.csv): ".encode())
            data = tcpclientsocket.recv(1024)
            if not data:
                continue
            file_path = data.decode()
            try:
                [x_train, y_train, x_test, y_test, data_range] = self.read_file(file_path)
            except FileNotFoundError as e:
                continue
            except PermissionError as e:
                continue
            break
        return [x_train, y_train, x_test, y_test, data_range]

    def start_parameter(self, net_config):
        parameter = {"net_config": net_config}
        para_json = json.dumps(parameter)
        HOST = '127.0.0.1'
        PORT = 6666

        clientsocket = socket(AF_INET, SOCK_STREAM)
        clientsocket.connect((HOST, PORT))
        while True:
            clientsocket.send(para_json.encode())
            print(para_json)
            break


## This is a container contains both parameter server and worker
## It is responsible for training, testing and prediction
class Container:
    def __init__(self, network):
        self.network = network

    def train(self, x_train, y_train, num_train):
        print("Training...")
        for e in range(num_train):
            loss = self.network.train(x_train, y_train)
            print("epoch " + str(e) + ", train loss: " + str(loss))

    def test(self, x_test, y_test):
        print("Testing...")
        y_predit = self.network.predict(x_test)
        loss = np.square(y_predit - y_test).sum() / x_test.shape[0]
        return loss

    def predict(self, x_input):
        return self.network.predict(x_input)


sever = Sever('', 10521, 5)
sever.start()
