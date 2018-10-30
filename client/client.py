'''创建客户端程序，向服务器传递数据'''
 
from socket import *
from time import ctime


def client():
    ## net config [1, 30, 0, 1]
    ## file path ../data/student_data.csv
    ## gre gpa rank
    ## gre range
    ## gpa range
    ## rank range
    HOST = '127.0.0.1'
    PORT = 10521
 
    clientsocket = socket(AF_INET,SOCK_STREAM)
    clientsocket.connect((HOST,PORT))
    while True:
        data = clientsocket.recv(1024)
        if not data:
            break
        print(data.decode())
        if data.decode() == "Training...":
            data = clientsocket.recv(1024)
            print(data.decode())
        if data.decode() == "Testing...":
            data = clientsocket.recv(1024)
            print(data.decode())
        data = input('Client>')
        if not data:
            break
        clientsocket.send(data.encode())
        
client()
