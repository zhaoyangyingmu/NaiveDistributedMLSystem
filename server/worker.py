import numpy as np
from util.network_elements import Network
from util.jsonsocket import *
from socket import *


class Worker:
    def __init__(self, HOST, PORT, listen_num):
        self.HOST = HOST
        self.PORT = PORT
        self.listen_num = listen_num

    def start(self):
        ADDR = (self.HOST, self.PORT)
        server_socket = JsonSocket(AF_INET, SOCK_STREAM)
        server_socket.bind(ADDR)
        server_socket.listen(self.listen_num)
        while True:
            print('Waiting for connecting ......')
            tcpclientsocket, addr = server_socket.accept()
            print('Connected by ', addr)

            ## get net config and x_train
            data = tcpclientsocket.recv()
            net_config = data["net_config"]
            x_train = data["x_train"]
            x_train = np.array(x_train)
            epoch = data["epoch"]

            network = Network(net_config)
            ## tell parameter, worker is ready
            tcpclientsocket.send({"mes": "worker is ready!!!"})

            for e in range(epoch):
                print("----------------------------------------------------------------------")
                print("epoch", e)
                ## receive weights and biases, then return layer_activations
                batch_size = 10
                for i in range(int(x_train.shape[0] / batch_size)):
                    data = tcpclientsocket.recv()
                    print("receive data:")
                    print(data)
                    weights = data["weights"]
                    biases = data["biases"]
                    weights = [np.array(weight) for weight in weights]
                    biases = [np.array(bias) for bias in biases]

                    layer_activations = network.forward(weights, biases, x_train[i:i+10, :])
                    layer_activations = [layer_activation.tolist() for layer_activation in layer_activations]
                    data = {"layer_activations" : layer_activations}
                    print("send data:")
                    print(data)
                    tcpclientsocket.send(data)
            tcpclientsocket.close()

