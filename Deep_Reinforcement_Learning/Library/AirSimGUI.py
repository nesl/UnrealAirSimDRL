# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tkinter as tk
import threading 
import time
import numpy as np
import subprocess
import cv2 
import sys
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util")
import ExcelLoader as XL
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

NUM_STATES = 13
IMG_HEIGHT = 28
IMG_WIDTH = 28
current_inertial_state = np.array(np.zeros(NUM_STATES))
sim_image = np.array(np.zeros((IMG_HEIGHT,IMG_WIDTH)))
lock = threading.Lock()

class QuadcopterInertialStateGUI(threading.Thread):
    
    def __init__(self, stateUpdateEvent, imageUpdateEvent):
        threading.Thread.__init__(self)
        self.stateUpdateEvent = stateUpdateEvent
        self.imageUpdateEvent = imageUpdateEvent
        
        self.lock = threading.Lock()
        self.start() # go to the run function
        
    def run(self):
        print("The Application is now running!")
        
        self.root = tk.Tk() # The GUI
        self.topFrame = tk.Frame(self.root) # Top of Gui 
        self.topFrame.pack(side = tk.TOP)
        self.middleFrame = tk.Frame(self.root)
        self.middleFrame.pack(side = tk.LEFT)
        self.rightFrame = tk.Frame(self.root, width = IMG_HEIGHT, height = IMG_WIDTH)
        self.rightFrame.pack(side = tk.RIGHT)
        #self.rightFrame.grid(row = 0, column = 0, padx = 10, pady = 2)
        self.bottomFrame = tk.Frame(self.root) # Bottom of GUI
        self.bottomFrame.pack(side = tk.BOTTOM)
        
        # Main Header of GUI
        self.MainWindowLabel = tk.Label(self.topFrame, text = "Quadcopter Inertial State Display")
        self.MainWindowLabel.pack()
        self.BottomLabel = tk.Label(self.bottomFrame, text = "Built for NESL AI")
        self.MainWindowLabel.pack()
        
        # Buttona on GUI
        self.button1 = tk.Button(self.topFrame, text = "Switch Modes", fg = "blue")
        #self.button2 = tk.Button(self.bottomFrame, text = "Launch Unreal", fg = "red")
        self.button1.pack()
        #self.button2.pack()
        
        # Labels for state variables
        self.label_posz = tk.Label(self.middleFrame, text = "Altitude:")
        self.label_velx = tk.Label(self.middleFrame, text = "Velocity X:")
        self.label_vely = tk.Label(self.middleFrame, text = "Velocity Y:")
        self.label_velz = tk.Label(self.middleFrame, text = "Velocity Z:")
        self.label_roll = tk.Label(self.middleFrame, text = "Roll:")
        self.label_pitch = tk.Label(self.middleFrame, text = "Pitch:")
        self.label_yaw = tk.Label(self.middleFrame, text = "Yaw:")
        self.label_accx = tk.Label(self.middleFrame, text = "Acceleration X:")
        self.label_accy = tk.Label(self.middleFrame, text = "Acceleration Y:")
        self.label_accz = tk.Label(self.middleFrame, text = "Acceleration Z:")
        self.label_rollrate = tk.Label(self.middleFrame, text = "Roll Rate:")
        self.label_pitchrate = tk.Label(self.middleFrame, text = "Pitch Rate:")
        self.label_yawrate = tk.Label(self.middleFrame, text = "Yaw Rate:")
        
        # Assemble into grid -- No need to pack if you are using grid
        self.label_posz.grid(row = 0, column = 0)
        self.label_velx.grid(row = 1, column = 0)
        self.label_vely.grid(row = 2, column = 0)
        self.label_velz.grid(row = 3, column = 0)
        self.label_roll.grid(row = 4, column = 0)
        self.label_pitch.grid(row = 5, column = 0)
        self.label_yaw.grid(row = 6, column = 0)
        self.label_accx.grid(row = 7, column = 0)
        self.label_accy.grid(row = 8, column = 0)
        self.label_accz.grid(row = 9, column = 0)
        self.label_rollrate.grid(row = 10, column = 0)
        self.label_pitchrate.grid(row = 11, column = 0)
        self.label_yawrate.grid(row = 12, column = 0)
        
        # Entries on GUI:
        self.entry_posz = tk.Entry(self.middleFrame, text = "Altitude:")
        self.entry_velx = tk.Entry(self.middleFrame, text = "Velocity X:")
        self.entry_vely = tk.Entry(self.middleFrame, text = "Velocity Y:")
        self.entry_velz = tk.Entry(self.middleFrame, text = "Velocity Z:")
        self.entry_roll = tk.Entry(self.middleFrame, text = "Roll:")
        self.entry_pitch = tk.Entry(self.middleFrame, text = "Pitch:")
        self.entry_yaw = tk.Entry(self.middleFrame, text = "Yaw:")
        self.entry_accx = tk.Entry(self.middleFrame, text = "Accel X:" )
        self.entry_accy = tk.Entry(self.middleFrame, text = "Accel Y:")
        self.entry_accz = tk.Entry(self.middleFrame, text = "Accel Z:")
        self.entry_rollrate = tk.Entry(self.middleFrame, text = "Roll Rate:")
        self.entry_pitchrate = tk.Entry(self.middleFrame, text = "Pitch Rate:")
        self.entry_yawrate = tk.Entry(self.middleFrame, text = "Yaw Rate:")

        # Entries in Grid
        self.entry_posz.grid(row = 0, column = 1)
        self.entry_velx.grid(row = 1, column = 1)
        self.entry_vely.grid(row = 2, column = 1)
        self.entry_velz.grid(row = 3, column = 1)
        self.entry_roll.grid(row = 4, column = 1)
        self.entry_pitch.grid(row = 5, column = 1)
        self.entry_yaw.grid(row = 6, column = 1)
        self.entry_accx.grid(row = 7, column = 1)
        self.entry_accy.grid(row = 8, column = 1)
        self.entry_accz.grid(row = 9, column = 1)
        self.entry_rollrate.grid(row = 10, column = 1)
        self.entry_pitchrate.grid(row = 11, column = 1)
        self.entry_yawrate.grid(row = 12, column = 1)
        
        # The Image Stramer From the simulator:
        self.image_panel = None
        self.cap = None
        
        # Run the inertial state listener thread
        self.t_states = threading.Thread(target = self.update_inertial_states, args = ())
        self.t_states.start() # Start Updater thread by setting the event
        # Run the Image Data Feed
        self.t_imgs = threading.Thread(target = self.update_image_feed)
        self.t_imgs.start()
        
        # Wait to launch
        time.sleep(.5)
        # Launch Application
        self.root.mainloop()
    
    def update_video_feed(self):
        while True:
            global sim_image
            self.imageUpdateEvent.wait()
            print("Updating GUI Image Feed")
            cv2image = cv2.cvtColor(sim_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = img)
            if self.image_panel is None: # Initialize the image panel
                self.image_panel = tk.Label(self.rightFrame,image=imgtk)
                self.image_panel.image = imgtk
                self.image_panel.pack(side="left", padx=10, pady=10)
            else:
                self.image_panel.configure(image = imgtk)
                self.image_panel.image = imgtk
                print("Frame Updated")
            self.imageUpdateEvent.clear()
    
    def update_image_feed(self):
        while True:
            global sim_image
            self.imageUpdateEvent.wait()
            print("Updating GUI Image Feed")
            #cv2image = cv2.cvtColor(sim_image, cv2.COLOR_BGR2RGB)
            with lock:
                img = Image.fromarray(sim_image)
            imgtk = ImageTk.PhotoImage(image = img)
            if self.image_panel is None: # Initialize the image panel
                self.image_panel = tk.Label(self.rightFrame,image=imgtk)
                self.image_panel.image = imgtk
                self.image_panel.pack(side="left", padx=10, pady=10)
            else:
                self.image_panel.configure(image = imgtk)
                self.image_panel.image = imgtk
                print("Frame Updated")
            self.imageUpdateEvent.clear()            
             
    def update_inertial_states(self):
        while True:
            self.stateUpdateEvent.wait()
            print('State Update Event Triggered!')
            global current_inertial_state
            
            with self.lock:
                self.entry_posz.delete(0,tk.END)
                self.entry_posz.insert(0, str(current_inertial_state[0]))
                self.entry_velx.delete(0,tk.END)
                self.entry_velx.insert(0, str(current_inertial_state[1]))
                self.entry_vely.delete(0,tk.END)
                self.entry_vely.insert(0, str(current_inertial_state[2]))
                self.entry_velz.delete(0,tk.END)
                self.entry_velz.insert(0, str(current_inertial_state[3]))
                self.entry_roll.delete(0,tk.END)
                self.entry_roll.insert(0, str(current_inertial_state[4]))
                self.entry_pitch.delete(0,tk.END)
                self.entry_pitch.insert(0, str(current_inertial_state[5]))
                self.entry_yaw.delete(0,tk.END)
                self.entry_yaw.insert(0, str(current_inertial_state[6]))
                self.entry_accx.delete(0,tk.END)
                self.entry_accx.insert(0, str(current_inertial_state[7]))
                self.entry_accy.delete(0,tk.END)
                self.entry_accy.insert(0, str(current_inertial_state[8]))
                self.entry_accz.delete(0,tk.END)
                self.entry_accz.insert(0, str(current_inertial_state[9]))
                self.entry_rollrate.delete(0,tk.END)
                self.entry_rollrate.insert(0, str(current_inertial_state[10]))
                self.entry_pitchrate.delete(0,tk.END)
                self.entry_pitchrate.insert(0, str(current_inertial_state[11]))
                self.entry_yawrate.delete(0,tk.END)
                self.entry_yawrate.insert(0, str(current_inertial_state[12]))
                self.stateUpdateEvent.clear()


def test_inertial_display(stateUpdateEvent):
        print("Incrementing State Vector")
        global current_inertial_state
        current_inertial_state += np.array(np.random.rand(NUM_STATES))
        stateUpdateEvent.set()


def test_videostream_gui_display(imgUpdateEvent):
    global sim_image
    if cap.isOpened():
        with lock:
            ret, sim_image = cap.read()
            sim_image = cv2.resize(sim_image, (IMG_HEIGHT, IMG_WIDTH))
            cv2.imshow("Video Feed", sim_image)
        cv2.waitKey(10)
        imgUpdateEvent.set()

def test_gui_img_display(imgUpdateEvent, img):
    global sim_image
    with lock:
        sim_image = np.reshape(img, (28,28))
        cv2.imshow("Video Feed", sim_image)
    cv2.waitKey(40)
    imgUpdateEvent.set()    
    
    
stateUpdateEvent = threading.Event()
imgUpdateEvent = threading.Event()
app = QuadcopterInertialStateGUI(stateUpdateEvent, imgUpdateEvent)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

i = 0
while(cv2.waitKey(15) != 27):
    # Load Image into cv frame
    test_inertial_display(stateUpdateEvent)
    test_videostream_gui_display(imgUpdateEvent)
    #test_gui_img_display(imgUpdateEvent, X_train[i])
    i += 1
    
#cv2.destroyAllWindows()  
#cap.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    