# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:17:01 2018

@author: natsn
"""

import sys, os
import socket
import queue
import threading
import pickle
import time


class TCPClient:
    def __init__(self, host = "127.0.0.1",
                 port = 5000,
                 buff_size = 1024):
        self.host = host # Local host almost always uses this IP address
        self.port = port
        self.message = {'data': None, 'ack':  None}
        self.buff_size = buff_size
        print("Attempting to connect to Host Server")
        self.s = socket.socket() # internal socket object
        self.s.connect((host, port)) # Bind the port to the local host
        self.isConnected = True

    def close(self):
        print('Closing Socket')
        self.s.close() 
        print('Socket Closed')
    
    def recv(self):
        try:
            if self.isConnected:
                data = self.s.recv(self.buff_size)
                #print(data)
                return pickle.loads(data)
            else:
                print("Cannot Receive: Client Not Connected")
                return None
        except EOFError:
            print("It appears as though the Host server has disconnected..")
            time.sleep(1)
    def recv_ack(self):
        if self.isConnected:
            data = self.s.recv(self.buff_size)
            self.message['ack'] = True
            self.send(self.message)
            return pickle.loads(data)
        else:
            print("Cannot Receive: Client Not Connected")
            return None
    
    def nack(self):
        if self.isConnected:
            data = self.s.recv(self.buff_size)
            self.message['ack'] = False
            self.send(self.message)
            self.isConnected = False
            print("Disconnected From Host")
            self.s.close()
            return pickle.loads(data)
        else:
            print("Not Connected! Cannot Nack!")
            return None
    
    def reconnect(self):
        if not self.isConnected:
            self.s = socket.socket()
            self.s.connect((self.host, self.port))
            self.isConnected = True
        else:
            print("Already Connected to Host!")
    
    def send(self, data):
        if self.isConnected:
            self.s.send(pickle.dumps(data))
        else:
            print("Cannot Send Data: Unconnected to Host!")


class SubscriberTCP(threading.Thread):
    def __init__(self, host="127.0.0.1",
                 port=5000,
                 BUFF_SIZE=1024,
                 QUEUE_SIZE = 10000,
                 callback=None): 
        threading.Thread.__init__(self)
                 
        self.host = host  # Local host almost always uses this IP address
        self.port = port
        self.BUFF_SIZE = BUFF_SIZE
        self.QUEUE_SIZE = QUEUE_SIZE
        self.callback = callback
        self.msg_queue = queue.Queue(self.QUEUE_SIZE)

        self.s = socket.socket()  # internal socket object
        self._connect_to_host()
        
    def _connect_to_host(self, attempts = 4):    
        for i in attempts:
            try:
                if not self.isConnected:
                    print("Connection Attempt #", i)
                    self.s.connect((self.host, self.port))  # Bind the port to the local host
                    self.isConnected = True
                    print("Connection Successful!")
                    self.start()
                    break
            except Exception:
                print("Unable to Connect to Socket!: Attempt ", i)
    
    def run(self):
        print("Launching Subscriber's Revieve Message")
        self._recv_msgs()
    
    def close(self):
        self.s.close()

    def _recv_msgs(self):
        try:
            if self.isConnected:
                data = self.s.recv(self.BUFF_SIZE)
                if data is not None:
                    data = pickle.loads(data)
                    self.msg_queue.put(data)
                    if self.callback is not None:
                        self.callback(data)
                    else:
                        print("I heard :", data)
        finally:
            print("It appears as though the server disconnected...")

    def _reconnect(self):
        if not self.isConnected:
            self.isConnected = True
            self.s = socket.socket()
            self._connect_to_host()
        else:
            print("Already Connected to Host!")

    # If Host has a listening channel, we could send to
    def send(self, data):
        if self.isConnected:
            self.s.send(pickle.dumps(data))
        else:
            print("Cannot Send Data: Unconnected to Host!")

def test_tcp_client():
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5005
    BUF_SIZE = 1024
    SLEEP_PERIOD = .00035
    client = TCPClient(host = TCP_IP, port = TCP_PORT, 
                buff_size = BUF_SIZE)
    try:
        while True:
            data = client.recv()
            if data is not None:
                print('controls ', data)
                time.sleep(SLEEP_PERIOD)
    except KeyboardInterrupt:
        print('Caught KeyBoard..Closing')
        client.close()
        time.sleep(2)
        print("Closed.")

def spin():
    SLEEP_PERIOD = .00035
    while True:
        time.sleep(SLEEP_PERIOD)

def callback(msg):
    print("Callback Function Heard: ", msg)


def test_subscribe():
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5005
    BUFF_SIZE = 1024
    QUEUE_SIZE = 10000
    print("Starting Subscriber")
    sub = SubscriberTCP(host=TCP_IP, port=TCP_PORT,
                           BUFF_SIZE=BUFF_SIZE, QUEUE_SIZE=QUEUE_SIZE, callback=callback)
    spin()

def tcp_turn_on_turn_on():
   TCP_IP = "127.0.0.1"
   TCP_PORT = 5005
   BUF_SIZE = 1024
   client = TCPClient(host = TCP_IP, port = TCP_PORT, 
                    buff_size = BUF_SIZE)
   data = None
   start_time = time.time()
   locked = False
   while True:
       if time.time() - start_time < 20: # seconds
           data = client.recv_ack()
           if data is not None:
               print("Xbox Controls: ", data)
           time.sleep(.00035)
       elif ((time.time() - start_time < 30) and (time.time() - start_time >= 20)):
           if not locked:
               client.nack() # Send Disconnect Signal
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
   
    

def main():
    #test_subscribe()
    test_tcp_client()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try: 
            print("Keyboard Escape!")
            sys.exit(0)
        except SystemError:
            os._exit(0)








