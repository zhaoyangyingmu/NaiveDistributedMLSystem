import numpy as np
from util.network_elements import Network
from util.jsonsocket import *
from socket import *


class ParameterServer:
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

            ## get net config, x_train, y_train, epoch
            data = tcpclientsocket.recv()
            net_config = data["net_config"]
            x_train = np.array(data["x_train"])
            y_train = np.array(data["y_train"])
            epoch = data["epoch"]
            network = Network(net_config)
            print("net config: " , net_config)
            print("epoch: ", epoch)

            ## communicate with worker
            self.communicate_with_worker(network, x_train, y_train, epoch)

            ## send weights and biases
            print("send weights and biases, training is over!!!")
            weights = network.get_weights()
            biases = network.get_biases()
            weights = [weight.tolist() for weight in weights]
            biases = [bias.tolist() for bias in biases]

            data = {"weights" : weights, "biases": biases}
            tcpclientsocket.send(data)
            tcpclientsocket.close()


    def communicate_with_worker(self, network, x_train, y_train, epoch):
        HOST = '127.0.0.1'
        PORT = 5555

        clientsocket = JsonSocket(AF_INET, SOCK_STREAM)
        clientsocket.connect((HOST, PORT))
        net_config = network.get_net_config()
        x_train = x_train.tolist()
        data = {"net_config": net_config, "x_train": x_train, "epoch": epoch}
        clientsocket.send(data)

        data = clientsocket.recv()
        print(data["mes"])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        for e in range(epoch):
            print("----------------------------------------------------------------------")
            print("epoch", e)
            batch_size = 10
            for i in range(int(x_train.shape[0]/batch_size)):
                weights = network.get_weights()
                biases = network.get_biases()
                weights = [weight.tolist() for weight in weights]
                biases = [bias.tolist() for bias in biases]
                data = {"weights": weights, "biases": biases}
                clientsocket.send(data)
                print("send data")
                print(data)

                data = clientsocket.recv()
                print("receive data")
                print(data)
                layer_activations = data["layer_activations"]
                layer_activations = [np.array(layer_activation) for layer_activation in layer_activations]
                network.train(x_train[i:i+10, :],y_train[i:i+10, :],layer_activations)
            ## print("layer_activations[0]", layer_activations[0])
        clientsocket.close()


