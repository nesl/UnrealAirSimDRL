# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:52:16 2018

@author: natsn
"""

import socket
import pickle
import time
import XboxListener
# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine

class UDPHost:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.buff_size = buff_size
        self.s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # internal socket object
        self.s.bind((host, port)) # Bind the port to the local host
        print("Server Started")
        self.addr = None
    
    def recv(self):
        data, self.addr = self.s.recvfrom(self.buff_size) # we can receive from multiple ports, since we are on a different one
        return self.run(pickle.loads(data))
    
    def run(self, data):
        if not data: # is None
            print("Got No Data!")
            return None
        print("From Client ", self.addr, "Data: ", data)
        #data = str(data).upper()
        return data
    
    def send(self, data):
        self.s.sendto(pickle.dumps(data), self.addr)
    
    def close(self):
        self.s.close()
    
    

if __name__ == "__main__":
    sample_rate = .00025
    host = UDPHost(buff_size = 1024)
    xbl = XboxListener.XBoxListener(sample_rate)
    unlocked = False
    while True:
        if not unlocked:
            data = host.recv()
            if type(data) == str:
                if data == "start":
                    print("Start Xbox Listener!")
                    xbl.init()
                    unlocked = True
                else:
                    data = "wrong"
                    host.send(data)
        if unlocked:
            data = xbl.get()
            if data is not None:
                print('controls ', data)
            host.send(data)
            time.sleep(.0004)
            




































































