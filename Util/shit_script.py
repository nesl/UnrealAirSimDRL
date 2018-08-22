# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:53:25 2018

@author: natsn
"""

import TCPClient
import time


if __name__ == "__main__":
    unlocked = False
    client = TCPClient.TCPClient(buff_size = 1024)
    while True:
        data = client.recv()
        if data is not None:
            print("Xbox Controls: ", data)
        time.sleep(.00035)




