'''创建客户端程序，向服务器传递数据'''
 
from jsocket import *
from socket import *


def client():
    ## net config [1, 30, 0, 1]
    ## file path ../data/student_data.csv
    ## gre gpa rank
    ## gre range
    ## gpa range
    ## rank range
    HOST = '127.0.0.1'
    PORT = 10521
 
    clientsocket = Jsocket(AF_INET,SOCK_STREAM)
    clientsocket.connect((HOST,PORT))
    while True:
        ## send net config and file path
        net_config = [3,1,1,2,-1]
        file_path = "..//data//student_data.csv"

        data = {"net_config": net_config, "file_path" : file_path}
        clientsocket.send(data)
        ## receive test result
        data = clientsocket.recv()
        print("test loss before training: " , data["test_before"])
        print("test loss after training: ", data["test_after"])
        ## after training finished, send data to predict
        while True:
            x_input = input("type in gre,gpa,rank for prediction:")
            if x_input == "close":
                data = {"x_input": "", "close": True}
                clientsocket.send(data)
                clientsocket.close()
                break
            data = {"x_input": x_input, "close": False}
            clientsocket.send(data)
            data = clientsocket.recv()
            predict_admission = data["predict_admission"]
            print("predict admission: ", predict_admission)
        break


client()
