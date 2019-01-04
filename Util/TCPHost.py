# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:20:03 2018

@author: natsn

modified by Kinree Aug 17 
"""

import traceback
import logging
import socket
import threading
import queue
import pickle
#from ExcelWriter import FileWriter
import time
import os, sys
import XboxListenerLinux
import numpy as np
lock = threading.Lock()
# Host: Binds to internal server, specifiy a port or keep default.
# Need to update run -- should run the specific Publishers message receive / transmit routine


class TCPHost(threading.Thread):
    def __init__(self, host="127.0.0.1",
                 port=5000,
                 BUFF_SIZE=1024,
                 MAX_SUBSCRIBERS=1):
        threading.Thread.__init__(self)

        self.host = host  # Local host almost always uses this IP address
        self.port = port
        self.cs = []
        self.addrs = []
        self.MAX_SUBSCRIBERS = MAX_SUBSCRIBERS
        self.current_client_count = 0
        self.current_client_id = 0
        self.BUFF_SIZE = BUFF_SIZE
        # internal socket object
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s.setsockopt(socket.IPPROTO_TCP, socket.SO_REUSEADDR, 1)
        self.s.bind((host, port))  # Bind the port to the local host
        print("Server Started")
        self.start()

    def check_add_addr(self, addr, c):
        if addr is not None:
            if addr not in self.addrs:
                # Check Num Listeners
                if self.current_client_count+1 <= self.MAX_SUBSCRIBERS:
                    print("New Connection from ", str(addr))
                    with lock:
                        self.current_client_count += 1
                        self.current_client_id += 1
                        self.cs.append(c)
                        self.addrs.append(addr)
                        return True
                else:
                    print("To Many Subscribers! Either up MAX_SUBSCRIBERS or connection will continue to be blocked")
        return False
    def recv(self, client_id=0):    
        if len(self.cs) > 0:
            data = self.cs[client_id].recv(self.BUFF_SIZE)
            if data is not None:
                #print(self.addrs[slave_num][1], datas)
                return pickle.loads(data)
        else:
            return None

    def run(self):
        print("Polling For Clients")
        self.poll_for_clients()

    def send(self, data):
        if len(self.cs) > 0:
            self.last_send_data_packet = data
            discons = []
            # Will send to all the registered tcp ports
            for i in range(len(self.cs)):
                try:
                    self.cs[i].send(pickle.dumps(data))
                except Exception:
                    print("Client ", i, " has disconnected")
                    discons.append(i)
                    self.current_client_count -= 1

            for d in discons:
                self.cs.pop(d)

    def send_ack(self, data):
        if len(self.cs) > 0:
            self.last_send_data_packet = data
            discons = []
            for i in range(len(self.cs)):

                # Will send to all the registered tcp ports
                self.cs[i].send(pickle.dumps(data))
                # Check for Ack or Nack from client
                client_data = self.recv(i)  # from client 'i'
                if client_data["ack"] != True:  # Take the guy out of the send to list
                    print("A client has disconnected")
                    discons.append(i)
                    self.current_client_count -= 1

            for d in discons:
                self.cs.pop(d)

    def poll_for_clients(self):
        # Listen for 1 connection at a time
        self.s.listen(self.MAX_SUBSCRIBERS)
        while True:
            if self.current_client_count <= self.MAX_SUBSCRIBERS:
                print("Initializing Clients..")
                c, addr = self.s.accept()
                if self.check_add_addr(addr, c):
                    print("Initialized ", self.current_client_count,
                        "/", self.MAX_SUBSCRIBERS, " Connections: ")
                    print("Connections Initialized!: ", c, addr)
                    print("Looking for new connections : ")
                    time.sleep(.5)
                else:
                    print("Did not initialize address...")

    def close(self):
        for c in self.cs:
            c.close()


# Peer To Peer
class PublisherTCP:
    def __init__(self, host="127.0.0.1",
                port=5000,
                BUFF_SIZE=1024,
                QUEUE_SIZE = 100,
                MAX_SUBSCRIPTIONS=1,
                service_func = None):

        self.host = host  # Local host almost always uses this IP address
        self.port = port
        self.c = None
        self.addr = None
        self.MAX_SUBSCRIPTIONS = MAX_SUBSCRIPTIONS
        self.QUEUE_SIZE = QUEUE_SIZE
        self.current_client_count = 0
        self.newest_client_id = 0
        self.client_id_tags = {}
        self.BUFF_SIZE = BUFF_SIZE
        self.service_func = service_func

        # internal socket object
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Listen for 1 connection at a time
        self.s.bind((host, port))  # Bind the port to the local host
        self.poll_thread = threading.Thread(target=self._poll_for_clients)
        print("TCP Host Server Started")
       
        # Message Connection Container
        self.message_subscriptions = {}
        self.open_message_threads = {}

        # Message Connection Indexes
        self.SUBSCRIPTION_ID_INDX = 0
        self.CONNECTION_INDX = 1
        self.ADDRESS_INDX = 2
        self.MSG_QUEUE_INDX = 3
        
        # initialize the messaging system
        self.SUBSCRIPTION_TYPES = ['sub', 'service']
        self.serviceTypes = ['data', 'routine']
        self.serviceCmds = ['ack', 'nack', 'resend']
        self.poll_thread.start()
        print("TCP Host Messaging Setup All Done!")

    def _poll_for_clients(self):
        self.s.listen(self.MAX_SUBSCRIPTIONS)
        while self.current_client_count <= self.MAX_SUBSCRIPTIONS:
            print("Initializing Clients..")
            #try:
            c, addr = self.s.accept()
            print("Found connection from new client")
            if self._check_add_addr_conn(c, addr):
                # setup channel
                print("Adding connection", (c,addr))
                self._msg_setup(c, addr)
                print("Initialized ", self.current_client_count,
                        "/", self.MAX_SUBSCRIPTIONS, " Connections: ")
                print("Connections Initialized!: ", c, addr)
                print("Looking for new connections : ")
                time.sleep(.5)
            #except Exception:
            #    print('Something wrong with stream of info')

    def _msg_setup(self, c, addr):
        msg_queue = queue.Queue(self.QUEUE_SIZE)
        # See what type of message service they want:
        service_choice = pickle.loads(c.recv(self.BUFF_SIZE))
        print("Subscriber Mode Selection: ", service_choice)
        if service_choice == self.SUBSCRIPTION_TYPES[0]:
            # Do Sub -- Thread of data from Pub to Sub
            client_id = self.newest_client_id
            subscription = (self.SUBSCRIPTION_TYPES[0], c, addr, msg_queue)
            self.message_subscriptions[client_id] = subscription
            print("New Initialized Subscription packet: ", subscription)
            msg_sub_thread = threading.Thread(target = self._msg_pub, args=(client_id,))
            self.open_message_threads[client_id] = msg_sub_thread
            print("Starting Publish to ", (c, addr), "Sub Nums:", self.current_client_count, self.client_id_tags)
            msg_sub_thread.start()
        elif service_choice == self.SUBSCRIPTION_TYPES[1]:
            # Do Sub -- Thread of data straight to it
            client_id = self.newest_client_id
            subscription = (self.SUBSCRIPTION_TYPES[0], c, addr, msg_queue, client_id)
            self.message_subscriptions[client_id] = subscription
            msg_sub_thread = threading.Thread(target = self._msg_service, args=(client_id,))
            self.open_message_threads[client_id] = msg_sub_thread
            msg_sub_thread.start()
        else:
            # Add for a resend
            pass

    def _close(self):
            self.c.close()

    def _check_add_addr_conn(self, c, addr):
        ADDRESS_OKAY = False
        if addr is not None:
            check_entries = np.array([c == x[self.ADDRESS_INDX][1] for _, x in self.message_subscriptions.items()], dtype=np.int)
            entries_already_exist = np.sum(check_entries)
            print("FOO FOO ", check_entries, entries_already_exist, c, addr)
            if  not entries_already_exist:
                print("FOO FOO")
                if self.current_client_count + 1 <= self.MAX_SUBSCRIPTIONS:
                    print("FOO FOO")
                    self.current_client_count += 1
                    self.newest_client_id += 1
                    self.client_id_tags[self.newest_client_id] = self.current_client_count
                    print("New Connection from ", str(addr))
                    ADDRESS_OKAY = True 
                else:
                    print("Too Many Subscribers - Either up MAX_SUBSCRIPTIONS or the new subscriber wont be added")
        return ADDRESS_OKAY

    # Messaging Routines
    def _msg_pub(self, client_id):
        while True:
            if not self.message_subscriptions[client_id][self.MSG_QUEUE_INDX].empty():
                if self.message_subscriptions[client_id][self.MSG_QUEUE_INDX].full():
                    self.message_subscriptions[client_id][self.MSG_QUEUE_INDX].get()
                try:
                    # Get Data to send
                    with lock:
                        raw_data = self.message_subscriptions[client_id][self.MSG_QUEUE_INDX].get()
                        print("Sending")
                        self.message_subscriptions[client_id][self.CONNECTION_INDX].send(pickle.dumps(raw_data))
                except Exception:
                    print('No Answer from: ', self.client_id_tags[client_id])
                    print('Disconnecting from: ', self.message_subscriptions[client_id]
                        [self.CONNECTION_INDX], self.message_subscriptions[client_id][self.ADDRESS_INDX])
                    # Cleanup:
                    #self.message_subscriptions[client_id][self.CONNECTION_INDX].send(pickle.dumps(raw_data)
                    del self.message_subscriptions[client_id]
                    #self.open_message_threads[client_id]._delete()
                    del self.client_id_tags[client_id]
                    self.current_client_count -= 1
                    print(self.message_subscriptions,
                        self.open_message_threads, self.client_id_tags)

    def _msg_service(self):
        while True:
            #1) Send Service CMD Options
            if self.c is not None and self.current_client_count > 0:
                try:
                    self.service_func()
                except Exception:
                    print('No Answer from: ', self.c, " or you didnt initialize both sides routines")
                    print('Disconnecting from: ', self.c)
                    self.c = None
                    self.addr = None
                    self.current_client_count -= 1    
    def publish(self, msg):
        if len(self.message_subscriptions) > 0:
            print(self.client_id_tags)
            for client_id in self.client_id_tags: # Grabs Key
                #print(msg, self.message_subscriptions)
                # Add thread pool
                with lock:
                    self.message_subscriptions[client_id][self.MSG_QUEUE_INDX].put(msg)


def test_publisher():
    XBOX_SAMPLE_PERIOD = .0025
    LOOP_DELAY = .025
    #TCP_IP = "127.0.0.1"
    TCP_IP = "192.168.1.212"
    TCP_PORT = 5007
    BUFF_SIZE = 1024
    QUEUE_SIZE = 100
    MAX_SUBSCRIBERS = 2
    host = PublisherTCP(host=TCP_IP, port=TCP_PORT,
                        BUFF_SIZE=BUFF_SIZE, QUEUE_SIZE=QUEUE_SIZE, 
                        MAX_SUBSCRIPTIONS=MAX_SUBSCRIBERS)
    #host = TCPHost(host = TCP_IP, port = TCP_PORT,
    #            buff_size = 1024, listen_for = LISTEN_FOR)
    print("Launch Xbox Listener")
    xbl = XboxListenerLinux.XBoxListener(XBOX_SAMPLE_PERIOD)
    xbl.init()
    print("Launched!")
    while True:
        data = xbl.get()
        #print("running..")
        if data is not None:
            #print('controls ', data)
            host.publish(data)
            time.sleep(LOOP_DELAY)


def test_tcphost():
    # Initialize TCP Host
    XBOX_SAMPLE_PERIOD = .00025
    LOOP_DELAY = .05
    TCP_IP = "127.0.0.1"
    TCP_PORT = 5005
    print(TCP_PORT)
    BUFF_SIZE = 10
    MAX_SUBSCRIBERS = 2
    host = TCPHost(host = TCP_IP, port = TCP_PORT, 
                   BUFF_SIZE=BUFF_SIZE, MAX_SUBSCRIBERS=MAX_SUBSCRIBERS)

    # Initialize XBox Controller
    xbl = XboxListenerLinux.XBoxListener(XBOX_SAMPLE_PERIOD)
    xbl.init()

    # Loop
    while True:
        data = xbl.get()
        #print("running..")
        if data is not None:
            #print('controls ', data)
            host.send(data)
            time.sleep(LOOP_DELAY)


def main():
    test_publisher()
    #test_tcphost()
        


if __name__ == "__main__":
    try:   
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
