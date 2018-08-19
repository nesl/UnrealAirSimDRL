# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:25:44 2018

@author: natsn
"""

import threading 
import time


class Subscriber:
    def __init__(self, name):
        self.name = name
        self.last_message = None
    def update(self, message):
        print("{} got message ".format(self.name, message))
        self.last_message = message


class Publisher:
    def __init__(self, events):
        self.subscribers = {event : dict() for event in events} # Seed dict
    def register(self, who, callback = None):
        if callback is None: # Goal is to register their method
            callback = getattr(who, 'update') #use the subscribers known update method
        self.subscribers(events)[who] = callback
    def get_subscribers(self, event):
        return self.subscribers[event]
    def unregister(self, event,who):
        del self.subscribers[event][who]
    def dispatch(self, message): # Go through subscribers and send messages
        for subscribers, callback in self.subscribers.items():
            callback(message)





a = Subscriber("Nat")
b = Subscriber("John")
c = Publisher()
