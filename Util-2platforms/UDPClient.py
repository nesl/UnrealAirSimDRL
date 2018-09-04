# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:52:16 2018

@author: natsn
"""

import socket
import pickle 
import time

# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine
# Server is a tuple of IP and PORT
# We can send arrays, dicts, strings, anything.
# We pickle before sending
class UDPClient:
    def __init__(self, server,
                 host = "127.0.0.1",
                 port = 5001,
                 buff_size = 1024):
        self.server = server
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.buff_size = buff_size
        self.s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # internal socket object
        self.s.bind((host, port)) # Bind the port to the local host
        print("Client Started")
        self.addr = None
    
    def recv(self):
        data, self.addr = self.s.recvfrom(self.buff_size)
        return self.run(pickle.loads(data))
    
    def run(self, data):
        if not data: # is None
            #print("Got No Data!")
            return None
        
        print("From Host ", data)
        #data = str(data).upper()
        return data
    
    def send(self, data):
        self.s.sendto(pickle.dumps(data), self.server) # Goes first to give the server its address
    
    def close(self):
        self.s.close()
    
    

if __name__ == "__main__":
    server = ("127.0.0.1", 5000)
    unlocked = False
    client = UDPClient(server, buff_size = 1024)
    data = input("Enter ''start'' to turn xbox listener on")
    while True:
        if not unlocked:
            data = input("Enter ''start'' to turn xbox listener on")
            client.send(data)
            data = client.recv()
            if data != "wrong":
                unlocked = True
            else:
                print("Incorrect key, re-enter")
        
        if unlocked:
            data = client.recv()
            if data is not None:
                print("Xbox Controls: ", data)
            time.sleep(.00035)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    