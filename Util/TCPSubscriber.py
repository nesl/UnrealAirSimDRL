# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 01:44:09 2018

@author: natsn
"""

import TCPClient
import Observer

class TCPSubscriber(TCPClient.TCPClient, Observer.Subscriber):
    
    def __init__(self, name, host, port, buff_size):
        TCPClient.TCPClient.__init__(self, host = host, port = port, buff_size = buff_size), 
        Observer.Subscriber.__init__(self, name = name)
        self.protocols = []
        
        self.msg_list = []
        self.new_msg = None
        # The packet of information wed like to send to the publisher to subscriber the update function as the callback
        self.protocol_update = {"who": name, "event": None}
    # Goal is to register the sub with the publisher
    def subscribe(self, event, protocol = "update"):
        self.protocol_update["event"] = event
        self.send(self.protocol_update) # Use the update function of the subscriber as the callback
        data = self.recv(self.buff_size)
        if data is not None:
            print("Subscribe Successful!")
            return 0
        else:
            print("subscribe failed..")
            return 1
        
        self.protocols.append(self.protocol_update)
        
    def update(self, message):
        print(self.name, " got message: ", message)
        self.new_msg = message
        self.msg_list.append(message)




               
if __name__ == "__main__":
    events = ["stream"]
    name = "tcpSub1"
    host = "127.0.0.1"
    port = 5000
    buff_size = 1024
    sub_lock = False
    tcpSub1 = TCPSubscriber(name, host, port, buff_size)
    
    while True:
        if not sub_lock:
            if tcpSub1.subscribe(events[0]) == 0:
                sub_lock = True
                print("The Subscriber was initialized correctly!")
            
            
        
        
        
        
    
    
    
