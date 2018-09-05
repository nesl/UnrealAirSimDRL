#!/usr/bin/env python

"""
A module for getting input from Microsoft XBox 360 controllers via the XInput library on Windows.

Adapted from Jason R. Coombs' code here:
http://pydoc.net/Python/jaraco.input/1.0.1/jaraco.input.win32.xinput/
under the MIT licence terms

Upgraded to Python 3
Modified to add deadzones, reduce noise, and support vibration
Only req is Pyglet 1.2alpha1 or higher:
pip install --upgrade http://pyglet.googlecode.com/archive/tip.zip 
"""

#import ctypes
import os
import sys
import time
import multiprocessing
import threading
import queue
import serial


class IMUQ:
    def __init__(self, buff_size = 10):
        self.imuq = queue.Queue()
        self.buff_size = buff_size
    def put(self, imuInput):
        if self.imuq.qsize() > self.buff_size:
            #print("Queue Full!")
            x = self.imuq.get()
            del x
        self.imuq.put(imuInput)
    def get(self):
        if self.imuq.qsize() < 1:
            #print("Queue Empty")
            return None
        else:
            return self.imuq.get()

def fmtFloat(n):
    return '{:6.3f}'.format(n)

def calibrate(N):
    calibrate_trial = N
    sum_pitch = 0
    sum_roll = 0
    sum_yaw = 0  
    sum_quatw = 0
    sum_quatx = 0
    sum_quaty = 0
    sum_quatz =0                       
    with serial.Serial('/dev/ttyACM0', 57600) as ser:
        line = ser.readline()
        for i in range(calibrate_trial):
            line= ser.readline()
            calib_data = [float(x) for x in str(line.decode('utf-8')).split(',')]
            sum_quatw += calib_data[7]
            sum_quatx += calib_data[8]
            sum_quaty += calib_data[9]
            sum_quatz += calib_data[10]
            sum_pitch += calib_data[11]
            sum_roll += calib_data[12]
            sum_yaw += calib_data[13]
        offset = [sum_quatw/(N*1.0), sum_quatx/(N*1.0), sum_quaty/(N*1.0), sum_quatz/(N*1.0),
                  sum_pitch/(N*1.0), sum_roll/(N*1.0), sum_yaw/(N*1.0)]
        return offset



# Globals
keys = ["TimeMS","accelX","accelY", "accelZ","gyroX", "gyroY", "gyroZ", 
        "quatW", "quatX", "quatY", "quatZ","pitch", "roll", "yaw","time"]

lock = threading.Lock()
imuq = IMUQ()


def sample_imu(dt,offset):
    start = time.time()
    #for Linux
    with serial.Serial('/dev/ttyACM0', 115200) as ser:
        line = ser.readline()
        while True:
            now = time.time() - start
            line = ser.readline()
            data = [float(x) for x in str(line.decode('utf-8')).split(',')]    

            imuInput = {keys[0]:data[0], keys[1]:data[1], 
                        keys[2]:data[2], keys[3]:data[3],
                        keys[4]:data[4], keys[5]:data[5], keys[6]:data[6], keys[7]:data[7]-offset[0],
                        keys[8]:data[8]-offset[1], keys[9]:data[9]-offset[2], keys[10]:data[10]-offset[3], keys[11]:data[11]-offset[4],
                        keys[12]:data[12]-offset[5], keys[13]:data[13]-offset[6], keys[14]:fmtFloat(now)}
            #print(imuInput)
            imuq.put(imuInput)
            time.sleep(dt)

    
        
        
class IMUListener(threading.Thread):
    def __init__(self, sample_rate):
        threading.Thread.__init__(self)
        #self.run = sample_first_joystick
        self.sample_rate = sample_rate
        self.time_start = time.time()
        
    def init(self):
        self.time_start = time.time()
        self.now = None
        self.start()
    
    def stop(self):
        self.join()
    
    def run(self):
        print("Start Calibrating!")
        offset = calibrate(500)
        print("Calibration is done!")
        sample_imu(self.sample_rate, offset)
    
    def get(self):
        global imuq
        self.now = time.time() - self.time_start
        return imuq.get()
        
if __name__ == '__main__':
    imul = IMUListener(0.001)
    imul.init()
    # while True:
    #     print(imul.get())

