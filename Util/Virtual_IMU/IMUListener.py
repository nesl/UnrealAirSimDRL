# -*- coding: utf-8 -*-
#!/usr/bin/env python

#import ctypes
import os
import sys
import time
import numpy as np
import multiprocessing
import threading
import queue
import serial

serial_port = '/dev/ttyACM0'
baud_rate = 57600  # 115200

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
    with serial.Serial(serial_port, baud_rate) as ser:  # or 115200 maybe 57600
    #with serial.Serial('COM9', 115200) as ser: -> Windows Version
        print('Turning off continous streaming')
        ser.write(b'#o0')
        print(ser.readline())
        print('Attempting to turn up the baud rate')
        #ser.write(b'r')
        print(ser.readline())
        #ser.write(b'r')
        print(ser.readline())
        #ser.write(b'r')
        print(ser.readline())
        print('Turning on continous streaming')
        ser.write(b'#o1')
        for i in range(50): # Flush
            print("Flushing..")
            print(ser.readline())
            time.sleep(.02)
        print("Done Flush")
        for i in range(calibrate_trial):
            print("Calibrating ", i, "/ ", calibrate_trial)
            line= ser.readline()
            calib_data = [float(x) for x in str(line.decode('utf-8')).split(',')]
            sum_quatw += calib_data[7]
            sum_quatx += calib_data[8]
            sum_quaty += calib_data[9]
            sum_quatz += calib_data[10]
            sum_pitch += calib_data[11]
            sum_roll += calib_data[12]
            sum_yaw += calib_data[13]
            print(sum_quatw, sum_quatx, sum_quaty)
        offset = [sum_quatw/(N*1.0), sum_quatx/(N*1.0), sum_quaty/(N*1.0), sum_quatz/(N*1.0),
                  sum_pitch/(N*1.0), sum_roll/(N*1.0), sum_yaw/(N*1.0)]
        return offset

# Globals
keys = ["TimeMS","accelX","accelY", "accelZ","gyroX", "gyroY", "gyroZ", 
        "quatW", "quatX", "quatY", "quatZ","pitch", "roll", "yaw","time"]

lock = threading.Lock()
imuq = IMUQ()


def sample_imu(hertz_frq, offset):
    start = time.time()
    #for Linux
    print("We did it..")
    time.sleep(1)
    with serial.Serial(serial_port, baud_rate) as ser:
    #with serial.Serial('COM9', 115200) as ser:
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
            time.sleep(1/hertz_frq)


class IMUListener(threading.Thread):
    def __init__(self, hertz_frq, dev_port = "/dev/ttyACM0", baud_rate = 57600, time_out = 1.2):
        threading.Thread.__init__(self)
        #self.run = sample_first_joystick
        self.hertz_frq = hertz_frq
        self.time_start = time.time()
        self.dev_port = dev_port 
        self.baud_rate = baud_rate
        self.time_out = time_out
        self.serial_port = serial.Serial(
            self.dev_port, self.baud_rate, timeout=self.time_out)
    
    def setup_imu(self, calibration_steps = 40):
        # 1) Check if we are even getting IMU readings..
        try:
            print("IMU is Functional: ", self.serial_port.readline())
        except:
            print("We Timed Out...Attempting to Restart the IMU")
            self.serial_port.write(b' ') # unpauses
            try:
                print("IMU is Functional: ", self.serial_port.readline())
            finally:
                print("Nothing Works, try reseting from Arduino Serial Monitor..")
                exit(1)
        print('Turning off continous streaming')
        ser.write(b'#o0')
        print("Adjusting IMU rate to specified sample rate -- rates between 0 - 100 Hz:")
        bins = np.array([1,10,20,30,40,50,60,70,80,90,100])
        t0 = time.time()
        print("Getting sample count over 5 seconds")
        datas = []
        while(time.time() - t0 < 5):
            datas.append(self.serial_port.readline())
            print("Count over", time.time() - t0 ,"is: ", len(datas))
        hertz_frq0 = np.digitize(len(datas) / 5.0, bins)
        self.hertz_frq = np.digitize(self.hertz_frq)
        print("Current Sample Rate is: ", hertz_frq0)
        print("Adjusting by: ", np.abs(self.hertz_frq - hertz_frq0))
        if self.hertz_frq > hertz_frq0:
            for i in range(int(np.max(self.hertz_frq) - np.max(hertz_frq0))):
                self.serial_port.write(b'r')
                print("Increasing sample time")
            print("The rate of the IMU is now: ", self.hertz_frq)
        else:
            for i in range(len(bins) - int(np.max(self.hertz_frq) + np.max(hertz_frq0))):
                self.serial_port.write(b'r')
                print("Increasing sample time")
            print("The rate of the IMU is now: ", self.hertz_frq)
        time.sleep(3)
        print("Start calibrating!")
        sum_pitch = 0
        sum_roll = 0
        sum_yaw = 0
        sum_quatw = 0
        sum_quatx = 0
        sum_quaty = 0
        sum_quatz = 0
        with serial.Serial(serial_port, baud_rate) as ser:  # or 115200 maybe 57600
            #with serial.Serial('COM9', 115200) as ser: -> Windows Version
            print('Turning on continous streaming')
            self.serial_port.write(b'#o1')
            for i in range(50):  # Flush
                print("Flushing..")
                print(self.serial_port.readline())
                time.sleep(.02)
            print("Done Flush")
            for i in range(calibration_steps):
                print("Calibrating ", i, "/ ", calibration_steps)
                line = self.serial_port.readline()
                calib_data = [float(x) for x in str(line.decode('utf-8')).split(',')]
                sum_quatw += calib_data[7]
                sum_quatx += calib_data[8]
                sum_quaty += calib_data[9]
                sum_quatz += calib_data[10]
                sum_pitch += calib_data[11]
                sum_roll += calib_data[12]
                sum_yaw += calib_data[13]
                print(sum_quatw, sum_quatx, sum_quaty)
            
            N = calibration_steps
            offset = [sum_quatw/(N*1.0), sum_quatx/(N*1.0), sum_quaty/(N*1.0), sum_quatz/(N*1.0),
                    sum_pitch/(N*1.0), sum_roll/(N*1.0), sum_yaw/(N*1.0)]
            print("Offsets found to be: ", offset)
            return offset

    def init(self):
        self.time_start = time.time()
        self.now = None
        self.start()
    
    def stop(self):
        self.join()
    
    def run(self):
        print("Start Calibrating!")
        offset = calibrate(50)
        print("Calibration is done!")
        sample_imu(self.hertz_frq, offset)
    
    def get(self):
        global imuq
        self.now = time.time() - self.time_start
        return imuq.get()



if __name__ == '__main__':
    sample_rate = 40
    imul = IMUListener(sample_rate)
    imul.init()
    time.sleep(10)
    print("Go!")
    while True:
        print(imul.get())
        time.sleep(1/sample_rate)
