# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 00:40:45 2018

@author: natsn
"""

from pynput import keyboard
import ExcelLoader as XLs
import cv2
import threading
import time

class KeyBoardListener():
    def __init__(self):
        # Collect events until released
        self.last_pressed = None
    
    # enter polling time in milliseconds
    def poll_keyboard(self, t):
        self.last_pressed = cv2.waitKey(t)
        return self.last_pressed
