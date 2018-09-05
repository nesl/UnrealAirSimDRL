
# -*- coding: utf-8 -*-

import time
import TCPClient
from ExcelWriter import FileWriter
import os
import KeyboardListener


client_count = 0

# listens to movements on the xbox controller through connecting to a TCP port
class IMUTCPClient(TCPClient.TCPClient):
    
    def __init__(self, host = "127.0.0.1", port = 5000, buff_size = 1024, 
                 isLeader = False, write_to_path = None, 
                 write_after = 500):
        TCPClient.TCPClient.__init__(self)
        
        self.write_after = write_after
        global client_count
        client_count += 1
        # if write_to_path is None:
        #     self.fWriter = FileWriter(os.getcwd() + "XboxTCPClient" + str(client_count) + ".csv")
        # else:
        #     self.fWriter = FileWriter(write_to_path)
        
        
        self.control_labels = ["TimeMS","accelX","accelY", "accelZ","gyroX", "gyroY", "gyroZ", 
                                   "quatW", "quatX", "quatY", "quatZ","pitch", "roll", "yaw", "time"]

        self.last_control = None
        self.control_dic = dict.fromkeys(self.control_labels, [])
            

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
        
        
    def recv_imu_update(self):
        if self.isConnected:
            controls = self.recv_ack()
            if controls is not None:
            #     self.format_controls(controls)
            #     if len(self.control_dic[self.control_labels[0]]) > self.write_after:
            #         self.fWriter.write_csv(self.control_dic)
            #         self.control_dic = dict.fromkeys(self.control_labels, []) # reset
                    
                self.last_control = controls
                return self.last_control
        else:
            self.last_control = None
            return self.last_control # is None
    



if __name__ == "__main__":
    #path = "IMUClient1.csv"
    #imuc = IMUTCPClient(write_to_path = path)
    imuc = IMUTCPClient()
    kbl = KeyboardListener.KeyboardListener(on_release = False, isPrintOnPress = True)  
    while True:
        control = imuc.recv_imu_update()
        if control is not None:
            print(control)
            #print(control["TimeMS"])
            #call visualization function
            
        key_info = kbl.get_last_key()
        if key_info is not None:
            if (key_info['key_val'] == '1'):
                print("NACKING")
                imuc.nack()
            if (key_info['key_val'] == '2'):
                imuc.reconnect()
        time.sleep(.0025)
