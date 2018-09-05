# -*- coding: utf-8 -*-
"""
kinree 2018.8.28
"""
from ExcelWriter import FileWriter
import IMUListener
import TCPHost
import os
import time

# listens to movements on the xbox controller through connecting to a TCP port
class IMUTCPHost(TCPHost.TCPHost):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 10, listen_for = 1,
                 write_to_path = None,
                 write_after = 500,
                 sample_rate = .001):
        TCPHost.TCPHost.__init__(self)
        
        self.commands = ["start", "ack", "nack", "new_leader"]
        self.write_after = write_after
        self.sample_rate = sample_rate
        
        # if write_to_path is None:
        #     self.fWriter = FileWriter(os.getcwd() + "IMUTCP.csv")
        # else:
        #     self.fWriter = FileWriter(write_to_path)
        
        self.control_labels = ["TimeMS","accelX","accelY", "accelZ","gyroX", "gyroY", "gyroZ", 
                               "quatW", "quatX", "quatY", "quatZ","pitch", "roll", "yaw", "time"]
        

        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
        
        
        self.imul = IMUListener.IMUListener(sample_rate)
        self.imul.init()
        print("IMU Listener On!")
    
    def format_controls(self, data):
        self.control_dic[self.control_labels[0]].append(data[self.control_labels[0]])
        self.control_dic[self.control_labels[1]].append(data[self.control_labels[1]])
        self.control_dic[self.control_labels[2]].append(data[self.control_labels[2]])
        self.control_dic[self.control_labels[3]].append(data[self.control_labels[3]])
        self.control_dic[self.control_labels[4]].append(data[self.control_labels[4]])
        self.control_dic[self.control_labels[5]].append(data[self.control_labels[5]])
        self.control_dic[self.control_labels[6]].append(data[self.control_labels[6]])
        self.control_dic[self.control_labels[7]].append(data[self.control_labels[7]])
        self.control_dic[self.control_labels[8]].append(data[self.control_labels[8]])
        self.control_dic[self.control_labels[9]].append(data[self.control_labels[9]])
        self.control_dic[self.control_labels[10]].append(data[self.control_labels[10]])
        self.control_dic[self.control_labels[11]].append(data[self.control_labels[11]])
        self.control_dic[self.control_labels[12]].append(data[self.control_labels[12]])
        self.control_dic[self.control_labels[13]].append(data[self.control_labels[13]])
        self.control_dic[self.control_labels[14]].append(data[self.control_labels[14]])


        
    def send_imu_update(self):
        controls = self.imul.get()
        if controls is not None:
            # self.format_controls(controls)
            # if len(self.control_dic[self.control_labels[0]]) > self.write_after:
            #     self.fWriter.write_csv(self.control_dic)
            #     self.control_dic = dict.fromkeys(self.control_labels, [])
            self.last_control = controls
        else:
            self.last_control = None
        self.send(controls)


if __name__ == "__main__":
    path = "IMUHost.csv"
    imuh = IMUTCPHost(write_to_path = path)
    while True:
        imuh.send_imu_update()
        if imuh.last_control is not None:
            print(imuh.last_control)
        time.sleep(.0025)
