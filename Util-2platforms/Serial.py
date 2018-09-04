# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:23:25 2018

@author: natsn
"""

import serial 
import time
import KeyboardListener
import XboxListener

sample_rate = .0025
s = serial.Serial(port='COM1', baudrate=115200, bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=sample_rate) 
isPrintOnPress = False
kbl = KeyboardListener.KeyboardListener(isPrintOnPress = isPrintOnPress,
                                        on_release = False)

xbl = XboxListener.XBoxListener(sample_rate)
xbl.init()

s.close()
s.open()
print(s.is_open)
print("Start Loop!")

while True:
    # Writing to Serial
    start = time.time()
    dataKB = kbl.get_last_key()
    dataXB = xbl.get()
    #print("Get Time: ", time.time() - start)
    if dataKB is not None:
        edata = str(dataKB).encode()
        s.write(edata)
    if dataXB is not None:
        print(dataXB)
        edata = str(dataXB).encode()
        s.write(edata)
    
    # Reading from Serial
    #print("hey")
    start2 = time.time()
    data = s.read(10)
    #print("Read Time: ", time.time() - start2)
    if data != b'':
        print("received serial data: ", data)
    data = data.decode()
    if data == 'm': # Message
        s.write(b'Python says Hi!')
    if data == 'q':
        print("Exiting!")
        exit(0)
        break
    time.sleep(.0025)
    
    

