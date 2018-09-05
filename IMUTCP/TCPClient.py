# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:17:01 2018

@author: natsn
"""

import socket
import pickle
import time


class TCPClient:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.message = {'data': None, 'ack':  None}
        self.buff_size = buff_size
        self.s = socket.socket() # internal socket object
        self.s.connect((host, port)) # Bind the port to the local host
        self.isConnected = True
        
    def recv_ack(self):
        if self.isConnected:
            data = self.s.recv(self.buff_size)
            self.message['ack'] = True
            self.send(self.message)
            return self.run(pickle.loads(data))
        else:
            return None
    
    def nack(self):
        if self.isConnected:
            data = self.s.recv(self.buff_size)
            self.message['ack'] = False
            self.send(self.message)
            self.isConnected = False
            print("Disconnected From Host")
            self.s.close()
            return self.run(pickle.loads(data))
        else:
            print("Not Connected! Cannot Nack!")
            return None
    
    def reconnect(self):
        if not self.isConnected:
            self.isConnected = True
            self.s = socket.socket()
            self.s.connect((self.host, self.port))
        else:
            print("Already Connected to Host!")
    
    def run(self, data):
        if not data: # is None
            #print("Got No Data!")
            return None
        
        #print("From Host! ", data)
        #data = str(data).upper()
        return data
    
    def send(self, data):
        if self.isConnected:
            self.s.send(pickle.dumps(data))
        else:
            print("Cannot Send Data: Unconnected to Host!")











