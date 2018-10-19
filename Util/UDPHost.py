# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:52:16 2018

@author: natsn
"""
import sys, os
import socket
import pickle
import time
import threading
import XboxListenerLinux
# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine

# Peer to Peer
# Three Threads: Poll, Send, Receieve
class UDPHost:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024,
                 max_num_subscribers = None):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.buff_size = buff_size
        self.subscriber_addresses = [] # Addresses of Subscribers
        self.isDonePollingForClients = False
        self.current_num_subscribers = 0
        self.s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # internal socket object
        self.s.bind((host, port))  # Bind the port to the local host
        #self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #print("Host Server Initialized")
        #self.poll_thread = threading.Thread(target = self.poll_for_clients)
        #self.poll_thread.start()
        #print("Polling For Clients Thread Started")

    def recv(self):
        data, _ = self.s.recvfrom(self.buff_size) # we can receive from multiple ports, since we are on a different one
        return pickle.loads(data)
    
    def send(self, data):
        #for i in range(len(self.subscriber_addresses)):
        self.s.sendto(pickle.dumps(data), (self.host, self.port))

    def close(self):
        self.s.close()
    

def main():
    sample_rate = .00025
    UDP_IP = '127.0.0.1'
    UDP_PORT = 5006
    host = UDPHost(UDP_IP, UDP_PORT, buff_size = 1024)
    xbl = XboxListenerLinux.XBoxListener(sample_rate)
    print("Start Xbox Listener!")
    xbl.init()
    while True:
        data = xbl.get()
        if data is not None:
            print('controls ', data)
            host.send(data)
        time.sleep(sample_rate)
     
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


