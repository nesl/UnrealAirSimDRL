# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:49:54 2018

@author: natsn
"""

import time
import TCPClient
from ExcelWriter import FileWriter
import os
import KeyboardListener

client_count = 0


# listens to movements on the xbox controller through connecting to a TCP port
class XboxControllerTCPClient(TCPClient.TCPClient):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 1024, 
                 isLeader = False, write_to_path = None,
                 write_after = 500):
        TCPClient.TCPClient.__init__(self)
        
        self.write_after = write_after
        global client_count
        client_count += 1
        if write_to_path is None:
            self.fWriter = FileWriter(os.getcwd() + "XboxTCPClient" + str(client_count) + ".csv")
        else:
            self.fWriter = FileWriter(write_to_path)
        
        self.control_labels = ["LA", "LAV", "LB", "LBIP", "Time"]
        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
            

    def format_controls(self, data):
        self.control_dic[self.control_labels[0]].append(data[self.control_labels[0]])
        self.control_dic[self.control_labels[1]].append(data[self.control_labels[1]])
        self.control_dic[self.control_labels[2]].append(data[self.control_labels[2]])
        self.control_dic[self.control_labels[3]].append(data[self.control_labels[3]])
        self.control_dic[self.control_labels[4]].append(data[self.control_labels[4]])
        
    def recv_controller_update(self):
        if self.isConnected:
            controls = self.recv_ack()
            if controls is not None:
                self.format_controls(controls)
                if len(self.control_dic[self.control_labels[0]]) > self.write_after:
                    self.fWriter.write_csv(self.control_dic)
                    self.control_dic = dict.fromkeys(self.control_labels, []) # reset
                    
                self.last_control = controls
                return self.last_control
        else:
            self.last_control = None
            return self.last_control # is None
    



if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__)) + "\\XboxClient1.csv"
    xbc = XboxControllerTCPClient(write_to_path = path)
    kbl = KeyboardListener.KeyboardListener(on_release = False, isPrintOnPress = True)  
    while True:
        control = xbc.recv_controller_update()
        if control is not None:
            print(control)
        key_info = kbl.get_last_key()
        if key_info is not None:
            if (key_info['key_val'] == '1'):
                print("NACKING")
                xbc.nack()
            if (key_info['key_val'] == '2'):
                xbc.reconnect()
        time.sleep(.0025)
        
    



















