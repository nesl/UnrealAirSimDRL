# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:49:57 2018

@author: natsn
"""


import socket
import pickle
from ExcelWriter import FileWriter
import XboxListener
import TCPHost
import time
import os




# listens to movements on the xbox controller through connecting to a TCP port
class XboxControllerTCPHost(TCPHost.TCPHost):
    
    def __init__(self, host, port, buff_size, listen_for = 1,
                 write_to_path = None,
                 write_after = 500,
                 sample_rate = .001):
        TCPHost.TCPHost.__init__(self)
        
        self.commands = ["start", "ack", "nack", "new_leader"]
        self.write_after = write_after
        self.sample_rate = sample_rate
        self.xbl = XboxListener.XBoxListener(sample_rate)
        
        if write_to_path is None:
            self.fWriter = FileWriter(os.getcwd() + "XboxTCP.csv")
        else:
            self.fWriter = FileWriter(write_to_path)
        
        self.control_labels = ["LA", "LAV", "LB", "LBIP", "Time"]
        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
        
        
    def start_stream(self):
        while self.recv(self.buff_size) != self.commands[0]:
            pass
        self.xbl.init()
        print("Xbox Listener On!")
            
    def poll_ack_nack(self):
        if self.recv() == self.commands[1]:
            pass
        elif self.recv() == self.commands[1]:
            pass
        else:
            pass
        
        self.send(self.commands[2])
    
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
                self.control_dic = dict.from_keys(self.control_labels, [])
            
            self.last_control = controls
            return self.last_control
        else:
            return self.last_control
    