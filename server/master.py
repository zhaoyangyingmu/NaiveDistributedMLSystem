'''创建服务器端程序，用来接收客户端传进的数据'''
 
from socket import *
import numpy as np
from network_elements import *
import json
from jsocket import Jsocket


class Master:
    def __init__(self, HOST, PORT, listen_num):
        self.HOST = HOST
        self.PORT = PORT
        self.listen_num = listen_num

    def start(self):
        ADDR = (self.HOST, self.PORT)
        server_socket = Jsocket(AF_INET, SOCK_STREAM)
        server_socket.bind(ADDR)
        server_socket.listen(self.listen_num)
        while True:
            print('Waiting for connecting ......')
            tcpclientsocket, addr = server_socket.accept()
            print('Connected by ', addr)

            ##one process
            while True:
                ## get net config and file path
                data = tcpclientsocket.recv()
                net_config = data["net_config"]
                file_path = data["file_path"]
                print("data received:")
                print("net config: ", net_config)
                print("file path: ", file_path)

                network = Network(net_config)

                ## get train file
                [x_train, y_train, x_test, y_test, data_range] = self.read_file(file_path)
                print("training data ready!")

                ## get a container
                container = Container(network)
                test_before = container.test(x_test, y_test)

                test_after = container.test(x_test, y_test)
                data = {"test_before": test_before, "test_after": test_after}
                tcpclientsocket.send(data)

                ## receive x_input and send output
                while True:
                    data = tcpclientsocket.recv()
                    if data["close"]:
                        tcpclientsocket.close()
                        break
                    x_input = data["x_input"]
                    print("data received: ")
                    print("x_input: " , x_input)
                    x_input = x_input.split(',')
                    x_input = [float(num) for num in x_input]
                    x_input = [x_input[i] / data_range[i] for i in range(len(x_input))]

                    predict_admission = (container.predict(x_input))[0,1]
                    print("predict_admission", predict_admission)
                    data = {"predict_admission": predict_admission}
                    tcpclientsocket.send(data)
                break

                ## begin training
                # try:
                tcpclientsocket.send("Training...".encode())
                container.train(x_train, y_train, 50)
                # except ValueError as e:
                #     print("Training failed, please try all again!")
                #     tcpclientsocket.send("You have put in wrong net config, please tr"
                #                          "y again! \nType in try again!\n".encode())
                #     continue
                # except Exception as e:
                #     print("Training failed, please try all again!")
                #     tcpclientsocket.send("You have put in wrong net config, please tr"
                #                          "y again! \nType in try again!\n".encode())
                #     continue

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


## This is a container contains both parameter server and worker
## It is responsible for training, testing and prediction
class Container:
    def __init__(self, network):
        self.network = network

    def train(self, x_train, y_train, num_train):
        HOST = '127.0.0.1'
        PORT = 6666

        clientsocket = Jsocket(AF_INET, SOCK_STREAM)
        clientsocket.connect((HOST, PORT))
        data = {"net_config": self.network.get_net_config(), "x_train": x_train.tolist(),
                "y_train": y_train.tolist(), "epoch": num_train}
        clientsocket.send(data)
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


sever = Master('', 10521, 5)
sever.start()
