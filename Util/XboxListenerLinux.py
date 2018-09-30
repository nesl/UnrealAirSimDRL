#!/usr/bin/env python

"""
A module for getting input from Microsoft XBox 360 controllers via the XInput library on Windows.

Adapted from Jason R. Coombs' code here:
http://pydoc.net/Python/jaraco.input/1.0.1/jaraco.input.win32.xinput/
under the MIT licence terms

Upgraded to Python 3
Modified to add deadzones, reduce noise, and support vibration
Only req is Pyglet 1.2alpha1 or higher:
pip install --upgrade http://pyglet.googlecode.com/archive/tip.zip 
"""

#import ctypes
import os
import sys
import time
from operator import itemgetter, attrgetter
#from itertools import count, starmap
from pyglet import event
#import xlwt
import multiprocessing
import threading
import queue

import xbox


class XboxQ:
    def __init__(self, buff_size = 10):
        self.xboxq = queue.Queue()
        self.buff_size = buff_size
    def put(self, boxInput):
        if self.xboxq.qsize() > self.buff_size:
            #print("Queue Full!")
            x = self.xboxq.get()
            del x
        self.xboxq.put(boxInput)
    def get(self):
        if self.xboxq.qsize() < 1:
            #print("Queue Empty")
            return None
        else:
            return self.xboxq.get()

def fmtFloat(n):
    return '{:6.3f}'.format(n)

# Globals
keys = ["leftX","leftY","rightX", "rightY", "A", "B", "X", "Y", 
        "dpadUp", "dpadDown", "dpadLeft", "dpadRight",
        "leftBumper","rightBumper","leftTrig","rightTrig",
        "Back","Guide","Start","Time"]

lock = threading.Lock()
xboxq = XboxQ()


def sample_first_joystick(dt):
    xbl = xbox.Joystick()

    start = time.time()
    #press B to exit
    while not xbl.Back():
        now = time.time() - start

        xboxInput = {keys[0]:fmtFloat(xbl.leftX()), keys[1]:fmtFloat(xbl.leftY()), 
                     keys[2]:fmtFloat(xbl.rightX()), keys[3]:fmtFloat(xbl.rightY()),
                     keys[4]:xbl.A(), keys[5]:xbl.B(), keys[6]:xbl.X(), keys[7]:xbl.Y(),
                     keys[8]:xbl.dpadUp(), keys[9]:xbl.dpadDown(), keys[10]:xbl.dpadLeft(), keys[11]:xbl.dpadRight(),
                     keys[12]:xbl.leftBumper(), keys[13]:xbl.rightBumper(), 
                     keys[14]:fmtFloat(xbl.leftTrigger()), keys[15]:fmtFloat(xbl.rightTrigger()),
                     keys[16]:xbl.Back(), keys[17]:xbl.Guide(), keys[18]:xbl.Start(), keys[19]:fmtFloat(now)}
        xboxq.put(xboxInput)
        time.sleep(dt)

    
        
        
class XBoxListener(threading.Thread):
    def __init__(self, sample_rate):
        threading.Thread.__init__(self)
        #self.run = sample_first_joystick
        self.sample_rate = sample_rate
        self.time_start = time.time()
        
    def init(self):
        self.time_start = time.time()
        self.now = None
        self.start()
    
    def stop(self):
        self.join()
    
    def run(self):
        sample_first_joystick(self.sample_rate)
    
    def get(self):
        #global LAST_BUTTON, LAST_BUTTON_IS_PRESSED, LAST_AXIS, LAST_AXIS_VALUE
        global xboxq
        self.now = time.time() - self.time_start
        return xboxq.get()
        #self.xboxq.put(xboxInput)


#################################
#if __name__ == "__main__":
#    xbl = XBoxListener(sample_rate = .02)
#    xbl.init()
#    while True:
#        print(xbl.get())
#        time.sleep(.02)
#        
#    
