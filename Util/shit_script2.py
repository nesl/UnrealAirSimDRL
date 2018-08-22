# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:53:25 2018

@author: natsn
"""

import time
import UDPHost


server = "127.0.0.1"
port = 5000
buff_size = 1024 # bytes

udp = UDPHost.UDPHost()

while True:
    data = udp.recv()
    print("From Client: ", data)
    time.sleep(.01)
    udp.send("Data received")
    
    
    
    





