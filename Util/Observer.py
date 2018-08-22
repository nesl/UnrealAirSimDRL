# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:25:44 2018

@author: natsn
"""

import threading 
import time
import socket
import TCPHost
import TCPClient

class Subscriber:
    def __init__(self, name):
        self.name = name
        self.last_message = None
    def update(self, message):
        print(self.name, " got message: ", message)
        self.last_message = message


class Publisher:
    def __init__(self, events, name):
        self.subscribers = {event : dict() for event in events} # Seed dict
        self.events = events
        self.name = name
    # Register to specific event
    def register(self, who, event = None, callback = None):
        if callback is None: # Goal is to register their method
            callback = getattr(who, 'update') #use the subscribers known update method
        if event is None:
            event = self.events[0]
        self.subscribers[event][who] = callback
    def get_subscribers(self, event):
        return self.subscribers[event]
    def unregister(self, event, who):
        del self.subscribers[event][who]
    def dispatch(self, message, event): # Go through subscribers and send messages
        for subscribers, callback in self.subscribers[event].items():
            callback(message)
            



if __name__ == "__main__":


#    sub1 = Subscriber("Sub1")
#    sub2 = Subscriber("Sub2")
#    sub3 = Subscriber("Sub3")
#    
#    pub1 = Publisher(events, "Pub1") # loads dictionary
#    
#    pub1.register(sub1, "start")
#    pub1.register(sub2, "start")
#    pub1.register(sub3, "stop")
#    
#    msg = "get it"
#    pub1.dispatch(msg, "start")
#    pub1.dispatch(msg, "stop")
    
    















