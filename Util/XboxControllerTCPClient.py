# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:49:54 2018

@author: natsn
"""

import socket
import pickle
import time
import TCPClient
from ExcelWriter import FileWriter
import os



# listens to movements on the xbox controller through connecting to a TCP port
class XboxControllerTCPClient(TCPClient.TCPClient):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 1024, 
                 isLeader = False, write_to_path = None,
                 write_after = 500):
        TCPClient.TCPClient.__init__(self)
        
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
    path = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util"
    xbc = XboxControllerTCPClient(write_to_path = path, isLeader = True)
    xbc.start_stream()
    while True:
        control = xbc.recv_controller_update()
        print(control)
        time.sleep(.001)
        
    



















