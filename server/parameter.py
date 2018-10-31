import numpy as np
from network_elements import *
from jsocket import *
from socket import *


class ParameterServer:
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

            ## get net config, x_train, y_train, epoch
            data = tcpclientsocket.recv()
            net_config = data["net_config"]
            x_train = np.array(data["x_train"])
            y_train = np.array(data["y_train"])
            epoch = data["epoch"]
            print("net config: " , net_config)
            print("epoch: ", epoch)
            print("x_train", x_train)
            print("y_train", y_train)



    def communicate_with_worker(self):
        pass


parameter_server = ParameterServer('', 6666, 5)
parameter_server.start()