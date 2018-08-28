# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:21:47 2018

@author: natsn
"""

# If you have 1 xbox clinet running, try this one as well and youll see the stream sent to both 

import time
import XboxControllerTCPClient
import KeyboardListener
import os

path = os.path.dirname(os.path.abspath(__file__)) + "\\XboxClient2.csv"
xbc = XboxControllerTCPClient.XboxControllerTCPClient(write_to_path = path)
kbl = KeyboardListener.KeyboardListener(on_release = False, isPrintOnPress = True)
while True:
    control = xbc.recv_controller_update()
    if control is not None:
        print(control)
    key_info = kbl.get_last_key()
    if key_info is not None:
        if (key_info['key_val'] == '3'):
            print("NACKING")
            xbc.nack()
        if (key_info['key_val'] == '4'):
            xbc.reconnect()
    time.sleep(.0025)