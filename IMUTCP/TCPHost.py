# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:20:03 2018

@author: natsn

modified by Kinree Aug 17 
"""

import socket
import threading
import pickle
from ExcelWriter import FileWriter
import time
import os

lock = threading.Lock()
# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine

class TCPHost(threading.Thread):
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024,
                 listen_for = 1):
        threading.Thread.__init__(self)
        
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.cs = []
        self.addrs = []
        self.listen_for = listen_for
        self.current_client_count = 0
        self.buff_size = buff_size
        self.last_rcv_data_packet = None
        self.last_send_data_packet = None
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # internal socket object
        self.s.bind((host, port)) # Bind the port to the local host
        print("Server Started")
        self.start()
        

    def check_add_addr(self, addr, c):
        if addr is not None:
            if addr not in self.addrs:
                print("New Connection from ", str(addr))
                with lock:
                    self.cs.append(c)
                    self.addrs.append(addr)
                    time.sleep(1)
    
    def recv(self, slave_num = 0):
        if len(self.cs) > 0:
            datas = self.cs[slave_num].recv(self.buff_size) #
            if datas is not None:
                #print(self.addrs[slave_num][1], datas)
                self.last_rcv_data_packet = datas
                return pickle.loads(datas)
        else:
            return None
    
    def run(self):
        print("Polling For Clients")
        self.poll_for_clients()
    
    def send(self, data):
        if len(self.cs) > 0:
            self.last_send_data_packet = data
            discons = []
            for i in range(len(self.cs)):
                self.cs[i].send(pickle.dumps(data)) # Will send to all the registered tcp ports
                
                # Check for Ack or Nack from client
                client_data = self.recv(i) # from client 'i'
                if client_data["ack"] != True: # Take the guy out of the send to list
                    print("A client has disconnected")
                    discons.append(i)
            for d in discons:
                self.cs.pop(d)
                    
    
    def poll_for_clients(self):
        self.s.listen(self.listen_for) # Listen for 1 connection at a time
        while True:
            print("Initializing Clients..")
            c, addr = self.s.accept()
            self.check_add_addr(addr, c)
            print("Initialized ", self.current_client_count , "/", self.listen_for, " Connections: ")
            print("Connections Initialized!: ", c, addr)
            print("Looking for new connections : ")
            time.sleep(.5)
        
        
    def close(self):
        for c in self.cs:
            c.close()
    



        
        
