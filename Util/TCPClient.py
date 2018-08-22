# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:17:01 2018

@author: natsn
"""

import socket
import pickle
import time
from ExcelWriter import FileWriter
import os

class TCPClient:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.buff_size = buff_size
        self.s = socket.socket() # internal socket object
        self.s.connect((host, port)) # Bind the port to the local host
        
        
    def recv(self):
        data = self.s.recv(self.buff_size)
        return self.run(pickle.loads(data))
    
    def run(self, data):
        if not data: # is None
            #print("Got No Data!")
            return None
        
        print("From Host! ", data)
        #data = str(data).upper()
        return data
    
    def send(self, data):
        self.s.send(pickle.dumps(data))
    
    def close(self):
        self.s.close()


# listens to movements on the xbox controller through connecting to a TCP port
class XboxControllerTCPClient(TCPClient):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 1024, 
                 isLeader = False, write_to_path = None,
                 write_after = 500):
        TCPClient.__init__(self)
        
        self.isLeader = isLeader
        self.commands = ["start", "ack", "nack", "new leader"]
        self.write_after = write_after
        
        if write_to_path is None:
            self.fWriter = FileWriter(os.getcwd() + "XboxTCP.csv")
        else:
            self.fWriter = FileWriter(write_to_path)
        
        self.control_labels = ["LA", "LAV", "LB", "LBIP", "Time"]
        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
            
    def start_stream(self):
        if self.isLeader:
            self.send(self.commands[0])
        else:
            print("Not Leader")
    
    def stop_stream(self):
        self.send(self.commands[2])
    
    def ack(self):
        if self.isLeader:
            self.send(self.commands[1])
    
    # Update
    def switch_leader(self):
        self.send(self.commands[3])
    
    def format_controls(self, data):
        self.control_dic[self.control_labels[0]] += data[self.control_labels[0]]
        self.control_dic[self.control_labels[1]] += data[self.control_labels[1]]
        self.control_dic[self.control_labels[2]] += data[self.control_labels[2]]
        self.control_dic[self.control_labels[3]] += data[self.control_labels[3]]
        self.control_dic[self.control_labels[4]] += data[self.control_labels[4]]
        
    def recv_controller_update(self):
        controls = self.recv(self.buff_size)
        if controls is not None:
            self.format_controls(controls)
            if len(self.control_dic[self.control_labels[0]]) > self.write_after:
                self.fWriter.write_csv(self.control_dic)
                self.control_dic = dict.from_keys(self.control_labels, []) # reset
                
            self.last_control = controls
            return self.last_control
        else:
            return self.last_control
    
    



if __name__ == "__main__":
    unlocked = False
    client = TCPClient(buff_size = 1024)
    data = None
    while True:
        if not unlocked:
            data = input("Enter ''start'' to turn xbox listener on")
            client.send(data)
            print("receiving!")
            data = client.recv()
            if data != "wrong":
                unlocked = True
                print("Start!")
            else:
                print("Incorrect key, re-enter")
        
        if unlocked:
            print("receiving!")
            data = client.recv()
            if data is not None:
                print("Xbox Controls: ", data)
            time.sleep(.00035)
    








