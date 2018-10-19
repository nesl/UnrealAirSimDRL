# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:52:16 2018

@author: natsn
"""
import os, sys
import socket
import pickle 
import time

# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine
# Server is a tuple of IP and PORT
# We can send arrays, dicts, strings, anything.
# We pickle before sending
class UDPClient:
    def __init__(self,
                host = "127.0.0.1",
                port = 5000,
                buff_size = 5000):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.buff_size = buff_size
        self.s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # internal socket object
        #self.s.bind((host, port)) # Bind the port to the local host
        #self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def recv(self):
        data, self.addr = self.s.recvfrom(self.buff_size)
        return pickle.loads(data)
    
    def send(self, data):
        self.s.sendto(pickle.dumps(data), (self.host, self.port)) # Goes first to give the server its address
    
    def close(self):
        self.s.close()
    
    
def main():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5006
    client = UDPClient(UDP_IP, UDP_PORT, buff_size=1024)
    print("Start UDP Client!")
    while True:
        data = client.recv()
        if data is not None:
            print("Xbox Controls: ", data)
        time.sleep(.00035)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
