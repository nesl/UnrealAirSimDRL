# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 00:40:45 2018

@author: natsn
"""

from pynput.keyboard import Key, Listener, KeyCode
import time
import queue

# Define Callback Functions
def on_press(key):
    print("Pressed Key: ", key)

def on_release(key):
    if key == Key.esc:
        exit(0)
    print("Released Key: ", key)

# Collect events from the keyboard listener
#with Listener(on_press = on_press, on_release = on_release) as listener:
    #listener.join()

class KeyBoardQueue:
    def __init__(self, max_buf_size = 20):
        self.kbq = queue.Queue()
        self.max_buf_size = max_buf_size
    def put(self, kbInput):
        if self.kbq.qsize() > self.max_buf_size:
            #print("KB Queue Full!")
            x = self.kbq.get()
            del x
        self.kbq.put(kbInput)
    def get(self):
        if self.kbq.qsize() < 1:
            #print("KB Queue Empty")
            return None
        else:
            return self.kbq.get()



class KeyboardListener(Listener):
    def __init__(self, on_press = None, 
                 on_release = None,
                 exit_key = Key.esc,
                 name = "Listener0",
                 isPrintOnPress = False,
                 max_buff_size = 8):
        if on_press is None:
            on_press = self.on_press
        if on_press == False:
            on_press = None
        
        if on_release is None:
            on_release = self.on_release
        if on_release == False:
            on_release = None
        
        Listener.__init__(self, on_press = on_press, on_release = on_release, daemon = True)
        
        self.keyQueue = KeyBoardQueue(max_buff_size)
        self.last_key = {}
        self.start_time = time.time()
        self.exit_key = exit_key
        self.name = name
        self.isPrintOnPress = isPrintOnPress
        self.start()
        
    def on_press(self, key):
        if key == self.exit_key:
            print("Ending Keyboard Listener " + self.name)
            self.join()
        if hasattr(key, 'char'):
            self.last_key = {'key': key, 'key_val': key.char, 'Time': time.time() - self.start_time, 'isPressed': True, "Name":self.name}
        else:
            self.last_key = {'key': key, 'key_val': None, 'Time': time.time() - self.start_time, 'isPressed': True, "Name":self.name}
        self.keyQueue.put(self.last_key)
        if self.isPrintOnPress:
            print("Pressed Key: ", key)
            
    
    def on_release(self, key):
        if key == self.exit_key:
            print("Ending Keyboard Listener " + self.name)
            self.join()
        if hasattr(key, 'char'):
            self.last_key = {'key': key, 'key_val': key.char, 'Time': time.time() - self.start_time, 'isPressed': True, "Name":self.name}
        else:
            self.last_key = {'key': key, 'key_val': None, 'Time': time.time() - self.start_time, 'isPressed': True, "Name":self.name}
        self.keyQueue.put(self.last_key)
        if self.isPrintOnPress:
            print("Released Key: ", key)
    def get_last_key(self):
        if not self.is_alive():
            print("Listener Process Stopped, Rebooting")
            self.join()
            print("Go Listen!")
            self.start()
        return self.keyQueue.get()


#if __name__ == "__main__":
#
#    kbl1 = KeyboardListener(exit_key = Key.f1, name = "KBL1")
#    kbl2 = KeyboardListener(exit_key = Key.f2, name = "KBL2")
#    
#    while True:
#        print("Last Key: ", kbl1.get_last_key())
#        print("Last Key: ", kbl2.get_last_key())
#        time.sleep(1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

