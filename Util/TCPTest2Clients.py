# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:53:25 2018

@author: natsn
"""

import TCPClient
import time

# Connects and disconnects depending on time
if __name__ == "__main__":
    client = TCPClient.TCPClient(buff_size = 1024, port = 37479)
    data = None
    start_time = time.time()
    locked = False
    while True:
        if time.time() - start_time < 20: # seconds
            data = client.recv_ack()
            if data is not None:
                print("Xbox Controls: ", data)
            time.sleep(.00035)
        elif ((time.time() - start_time <30) and (time.time() - start_time >= 20)):
            if not locked:
                client.nack()
                locked = True
                print("disconnected!")
        else:
            if locked:
                client.reconnect()
                locked = False
            data = client.recv_ack()
            if data is not None:
                print("Xbox Controls: ", data)
        time.sleep(.00035)
