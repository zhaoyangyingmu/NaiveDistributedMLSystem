import socket
import json
import struct

import _socket


class Jsocket(socket.socket):

    def __init__(self, family=-1, type=-1, proto=0, fileno=None):
        super(Jsocket, self).__init__(family, type, proto, fileno)

    def recv(self):

        # 接收报头长度
        res = super(Jsocket, self).recv(4)
        if len(res) != 4:
            return {}
        # 对报头长度解压
        lens = struct.unpack('i', res)
        header_size = lens[0]
        # 接收报头长度的内容
        header_bytes = super(Jsocket, self).recv(header_size).decode('utf-8')
        # 对报头字典进行反序列化
        header_json = json.loads(header_bytes)
        # 获取数据长度
        data_size = header_json['total_size']

        # 根据报头数据长度对数据进行接收
        recv_size = 0
        total_data = b''
        while recv_size < data_size:
            left_size = data_size - recv_size
            data_recv = b''
            if data_size - recv_size > 1024:
                
                left_size = 1024
            
            data_recv = super(Jsocket, self).recv(left_size)
            recv_size += left_size
            total_data += data_recv

        recvJson = total_data.decode('utf-8')
        return json.loads(recvJson)

    def send(self, sendDict):
        sendJson = json.dumps(sendDict)

        sendFile = sendJson.encode('utf-8')

        # 定制报头
        header_dict = {
            'total_size': len(sendFile),
            'filename': None
        }
        header_bytes = json.dumps(header_dict).encode('utf-8')
        # 发送报头长度
        header_len = struct.pack('i', len(header_bytes))
        super(Jsocket, self).send(header_len)
        # 发送报头
        super(Jsocket, self).send(header_bytes)
        # 发送数据部分
        super(Jsocket, self).send(sendFile)

    def accept(self):
        fd, addr = super(Jsocket, self)._accept()
        jsock = Jsocket(super(Jsocket, self).family, super(Jsocket, self).type, super(Jsocket, self).proto, fileno=fd)
        # Issue #7995: if no default timeout is set and the listening
        # socket had a (non-zero) timeout, force the new socket in blocking
        # mode to override platform-specific socket flags inheritance.
        if _socket.getdefaulttimeout() is None and super(Jsocket, self).gettimeout():
            jsock.setblocking(True)
        return jsock, addr


