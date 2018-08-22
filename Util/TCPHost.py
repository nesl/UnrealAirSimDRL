# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:20:03 2018

@author: natsn
"""

import socket
import pickle
import XboxListener
from ExcelWriter import FileWriter
import time
import os

# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine

class TCPHost:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024,
                 listen_for = 1):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.cs = []
        self.addrs = []
        self.listen_for = listen_for
        self.buff_size = buff_size
        self.last_rcv_data_packet = None
        self.last_send_data_packet = None
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # internal socket object
        self.s.bind((host, port)) # Bind the port to the local host
        print("Server Started")
        self.init() # Connects to the host port automatically
    
    # Can call to re connect to the port
    def init(self):
        print("Initializing Connections: ")
        self.s.listen(self.listen_for) # Listen for 1 connection at a time
        for i in range(self.listen_for):
            c, addr = self.s.accept()
            self.check_add_addr(addr, c)
            print("Initialized ", i+1, "/2 Connections: ")
        print("Connections Initialized!")
        time.sleep(1)
        
    def check_add_addr(self, addr, c):
        if addr is not None:
            if addr not in self.addrs:
                print("New Connection from ", str(addr))
                self.cs.append(c)
                self.addrs.append(addr)
                time.sleep(1)
    
    def recv(self, slave_num = 0):
        datas = self.cs[slave_num].recv(self.buff_size) #
        print(self.addrs[slave_num][1], datas)
        self.last_rcv_data_packet = datas
        print(datas)
        return self.run(pickle.loads(datas))
    
    def run(self, datas):
        if not datas: # is None
            print("Got No Data!")
            return None
        
        print("From User", datas)
        #data = str(data).upper()
        return datas
    
    def send(self, data):
        self.last_send_data_packet = data
        for c in self.cs:
            c.send(pickle.dumps(data)) # Will send to all the registered tcp ports
    
    def close(self):
        for c in self.cs:
            c.close()
    



if __name__ == "__main__":
    sample_rate = .00025
    msg_start_slave = 0
    listen_for = 2
    host = TCPHost(buff_size = 1024, listen_for = listen_for)
    xbl = XboxListener.XBoxListener(sample_rate)
    unlocked = False
    while True:
        if not unlocked:
            print("Receiving..")
            data = host.recv(msg_start_slave)
            print(data)
            if type(data) == str:
                if data == "start":
                    print("Start Xbox Listener!")
                    xbl.init()
                    unlocked = True
                else:
                    data = "wrong password"
                    host.send(data)
        if unlocked:
            data = xbl.get()
            if data is not None:
                print('controls ', data)
                host.send(data)
            time.sleep(.0004)