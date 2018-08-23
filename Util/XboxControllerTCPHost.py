# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:49:57 2018

@author: natsn
"""



from ExcelWriter import FileWriter
import XboxListener
import TCPHost
import os
import time


# listens to movements on the xbox controller through connecting to a TCP port
class XboxControllerTCPHost(TCPHost.TCPHost):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 10, listen_for = 1,
                 write_to_path = None,
                 write_after = 500,
                 sample_rate = .001):
        TCPHost.TCPHost.__init__(self)
        
        self.commands = ["start", "ack", "nack", "new_leader"]
        self.write_after = write_after
        self.sample_rate = sample_rate
        
        if write_to_path is None:
            self.fWriter = FileWriter(os.getcwd() + "XboxTCP.csv")
        else:
            self.fWriter = FileWriter(write_to_path)
        
        self.control_labels = ["LA", "LAV", "LB", "LBIP", "Time"]
        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
        
        self.xbl = XboxListener.XBoxListener(sample_rate)
        self.xbl.init()
        print("Xbox Listener On!")
    
    def format_controls(self, data):
        self.control_dic[self.control_labels[0]].append(data[self.control_labels[0]])
        self.control_dic[self.control_labels[1]].append(data[self.control_labels[1]])
        self.control_dic[self.control_labels[2]].append(data[self.control_labels[2]])
        self.control_dic[self.control_labels[3]].append(data[self.control_labels[3]])
        self.control_dic[self.control_labels[4]].append(data[self.control_labels[4]])
        
    def send_controller_update(self):
        controls = self.xbl.get()
        if controls is not None:
            self.format_controls(controls)
            if len(self.control_dic[self.control_labels[0]]) > self.write_after:
                self.fWriter.write_csv(self.control_dic)
                self.control_dic = dict.fromkeys(self.control_labels, [])
            self.last_control = controls
        else:
            self.last_control = None
        self.send(controls)






if __name__ == "__main__":
    path = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util\\XboxHost.csv"
    xbh = XboxControllerTCPHost(write_to_path = path)
    while True:
        xbh.send_controller_update()
        if xbh.last_control is not None:
            print(xbh.last_control)
        time.sleep(.0025)
        
    














