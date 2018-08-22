# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 01:40:10 2018

@author: natsn
"""

import Observer
import TCPHost
import time


class TCPPublisher(TCPHost.TCPHost, Observer.Publisher):
    
    def __init__(self, events , name, host, port, buff_size):
        TCPHost.TCPHost.__init__(self, host = host, port = port, buff_size = buff_size), 
        Observer.Publisher.__init__(self, events = events, name = name)
        self.protocols = []
        
        self.msg_list = []
        self.new_msg = None
        # The packet of information wed like to send to the publisher to subscriber the update function as the callback
        self.protocol_update = {"who": name, "event": None}
    
    # Get subscribers awaiting a request
    def add_subscribers(self):
        data = self.recv() # data = the subscribers info 
        if data is not None:
            if list(data.keys()) == list(self.protocol_update.keys()):
                self.register(data["who"], data["event"])
                print("register successful! ", data["who"])
                self.protocol_update["who"] = data["who"]
                self.protocol_update["event"] = data["event"]
                self.protocols.append(self.protocol_update)
        else:
            print("No new subscribers")
    
    def subscriber_count(self):
        return len(self.protocols)
                
if __name__ == "__main__":
    events = ["stream"]
    name = "tcpPub1"
    host = "127.0.0.1"
    port = 5000
    buff_size = 1024
    
    tcpPub1 = TCPPublisher(events = events, name = name, host = host, port = port, buff_size = buff_size)
    
    while True:
        tcpPub1.add_subscribers()
        if tcpPub1.subscriber_count() != 0:
            tcpPub1.dispatch()
        time.sleep(.5)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
                
                
                
                