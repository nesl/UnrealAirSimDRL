# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 10:55:37 2018

@author: natsn
"""

import tkinter as tk
from tkinter.ttk import Label, Frame, Entry, Notebook, Combobox
import threading 
import multiprocessing
import time
import numpy as np
import sys
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util")
import ExcelLoader as XL
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import  axes3d,Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import cm
from PIL import ImageTk, Image


class QuadcopterGUI(multiprocessing.Process):
    
    def __init__(self, statePipe, imagePipe, vehicle_names, num_video_feeds, isRGB = True):
        multiprocessing.Process.__init__(self)
        self.statePipe = statePipe
        self.imagePipe = imagePipe
        self.vehicle_names = vehicle_names
        self.num_video_feeds = num_video_feeds
        self.isRGB = isRGB
    
        self.start() # go to the run function
        
    def run(self):
        print("The Application is now running!")
        
        self.root = tk.Tk() # The GUI
        self.root.title("Quadcopter Unreal GUI")
        self.nb = Notebook(self.root)
        
        # Add Main Tab
        self.StateFrame = Frame(self.nb) # Top of Gui 
        self.nb.add(self.StateFrame, text = "Vehicle State")
        
        # Get Notebook Gridded
        self.nb.grid(row = 0, column = 0)
        
        # Configure Video Tab
        self.VideoFrame = Frame(self.nb)
        self.nb.add(self.VideoFrame, text = "Video Feed")
        
        # Configure Plotting Tab
        self.PlottingFrame = Frame(self.nb)
        self.nb.add(self.PlottingFrame, text = "Graphics")
        self.fig = plt.figure(1, figsize= (4,4))
        self.ax = Axes3D(self.fig)
        self.ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.PlottingFrame)
        self.canvas.get_tk_widget().grid(row = 1, column = 0)
        
        
        # For Switch Vehicle Feeds
        self.current_vehicle_feed = tk.StringVar(self.StateFrame) # Linked to current vehicle choice
        self.switch_vehicle_feed = Combobox(self.StateFrame, textvariable = self.current_vehicle_feed)
        self.switch_vehicle_feed['values'] = self.vehicle_names
        self.switch_vehicle_feed.grid(row = 0, column = 3)
        self.switch_vehicle_feed.current(0)
        
        
        # Labels for state variables
        self.label_posx = Label(self.StateFrame, text = "PosX:")
        self.label_posy = Label(self.StateFrame, text = "PosY:")
        self.label_posz = Label(self.StateFrame, text = "PosZ:")
        self.label_velx = Label(self.StateFrame, text = "Vel X:")
        self.label_vely = Label(self.StateFrame, text = "Vel Y:")
        self.label_velz = Label(self.StateFrame, text = "Vel Z:")
        self.label_accx = Label(self.StateFrame, text = "Acc X:")
        self.label_accy = Label(self.StateFrame, text = "Acc Y:")
        self.label_accz = Label(self.StateFrame, text = "Acc Z:")
        self.label_roll = Label(self.StateFrame, text = "Roll:")
        self.label_pitch = Label(self.StateFrame, text = "Pitch:")
        self.label_yaw = Label(self.StateFrame, text = "Yaw:")
        self.label_rollrate = Label(self.StateFrame, text = "Roll Rate:")
        self.label_pitchrate = Label(self.StateFrame, text = "Pitch Rate:")
        self.label_yawrate = Label(self.StateFrame, text = "Yaw Rate:")
        self.label_angaccx = Label(self.StateFrame, text = "AngAcc X:")
        self.label_angaccy = Label(self.StateFrame, text = "AngAcc Y:")
        self.label_angaccz = Label(self.StateFrame, text = "AngAcc Z:")
        
        
        # Assemble into grid -- No need to pack if you are using grid
        self.label_posx.grid(row = 0, column = 0)
        self.label_posy.grid(row = 1, column = 0)
        self.label_posz.grid(row = 2, column = 0)
        self.label_velx.grid(row = 3, column = 0)
        self.label_vely.grid(row = 4, column = 0)
        self.label_velz.grid(row = 5, column = 0)
        self.label_accx.grid(row = 6, column = 0)
        self.label_accy.grid(row = 7, column = 0)
        self.label_accz.grid(row = 8, column = 0)
        self.label_roll.grid(row = 9, column = 0)
        self.label_pitch.grid(row = 10, column = 0)
        self.label_yaw.grid(row = 11, column = 0)
        self.label_rollrate.grid(row = 12, column = 0)
        self.label_pitchrate.grid(row = 13, column = 0)
        self.label_yawrate.grid(row = 14, column = 0)
        self.label_angaccx.grid(row = 15, column = 0)
        self.label_angaccy.grid(row = 16, column = 0)
        self.label_angaccz.grid(row = 17, column = 0)
        
        # Entries on GUI:
        # Labels for state variables
        self.entry_posx = Entry(self.StateFrame, text = "PosX:")
        self.entry_posy = Entry(self.StateFrame, text = "PosY:")
        self.entry_posz = Entry(self.StateFrame, text = "PosZ:")
        self.entry_velx = Entry(self.StateFrame, text = "Vel X:")
        self.entry_vely = Entry(self.StateFrame, text = "Vel Y:")
        self.entry_velz = Entry(self.StateFrame, text = "Vel Z:")
        self.entry_accx = Entry(self.StateFrame, text = "Acc X:")
        self.entry_accy = Entry(self.StateFrame, text = "Acc Y:")
        self.entry_accz = Entry(self.StateFrame, text = "Acc Z:")
        self.entry_roll = Entry(self.StateFrame, text = "Roll:")
        self.entry_pitch = Entry(self.StateFrame, text = "Pitch:")
        self.entry_yaw = Entry(self.StateFrame, text = "Yaw:")
        self.entry_rollrate = Entry(self.StateFrame, text = "Roll Rate:")
        self.entry_pitchrate = Entry(self.StateFrame, text = "Pitch Rate:")
        self.entry_yawrate = Entry(self.StateFrame, text = "Yaw Rate:")
        self.entry_angaccx = Entry(self.StateFrame, text = "AngAcc X:")
        self.entry_angaccy = Entry(self.StateFrame, text = "AngAcc Y:")
        self.entry_angaccz = Entry(self.StateFrame, text = "AngAcc Z:")

        # Entries in Grid
        self.entry_posx.grid(row = 0, column = 1)
        self.entry_posy.grid(row = 1, column = 1)
        self.entry_posz.grid(row = 2, column = 1)
        self.entry_velx.grid(row = 3, column = 1)
        self.entry_vely.grid(row = 4, column = 1)
        self.entry_velz.grid(row = 5, column = 1)
        self.entry_accx.grid(row = 6, column = 1)
        self.entry_accy.grid(row = 7, column = 1)
        self.entry_accz.grid(row = 8, column = 1)
        self.entry_roll.grid(row = 9, column = 1)
        self.entry_pitch.grid(row = 10, column = 1)
        self.entry_yaw.grid(row = 11, column = 1)
        self.entry_rollrate.grid(row = 12, column = 1)
        self.entry_pitchrate.grid(row = 13, column = 1)
        self.entry_yawrate.grid(row = 14, column = 1)
        self.entry_angaccx.grid(row = 15, column = 1)
        self.entry_angaccy.grid(row = 16, column = 1)
        self.entry_angaccz.grid(row = 17, column = 1)
        
        # The Image Stramer From the simulator:
        self.VideoFeeds = [None for i in range(self.num_video_feeds)]

        
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
    

    def update_image_feed(self):
        scalar = 3
        col_count = 0
        row_count = 0
        if not self.isRGB:
            scalar = 1
        
        while True:
            vehicle_sim_images = self.imagePipe.recv()
            sim_images = vehicle_sim_images[self.current_vehicle_feed.get()]
            print("Updating GUI Image Feed")
            for i in range(self.num_video_feeds):
                sim_image = sim_images[:,:,scalar*i:scalar*(i+1)]
                if scalar == 1:
                    sim_image = np.reshape(sim_image, (sim_image.shape[0], sim_image.shape[1]))
                if ((i % 3) == 0):
                    col_count = 0
                    row_count += 1
                img = Image.fromarray(sim_image)
                imgtk = ImageTk.PhotoImage(image = img)
                if self.VideoFeeds[i] is None: # Initialize the image panel
                    self.VideoFeeds[i] = Label(self.VideoFrame, image=imgtk)
                    self.VideoFeeds[i].image = imgtk
                    self.VideoFeeds[i].grid(row = row_count, column = col_count)
                else:
                    self.VideoFeeds[i].configure(image = imgtk)
                    self.VideoFeeds[i].image = imgtk
                col_count += 1
            col_count = 0
            row_count = 0
            print("Feed Update")
            
             
    def update_inertial_states(self):
        while True:
            current_inertial_states = self.statePipe.recv()
            print(current_inertial_states)
            current_inertial_state = current_inertial_states[self.current_vehicle_feed.get()]
            print('State Update Event Triggered!')
            print(current_inertial_states, current_inertial_state)
            self.entry_posx.delete(0,tk.END)
            self.entry_posx.insert(0, str(current_inertial_state[0]))
            self.entry_posy.delete(0,tk.END)
            self.entry_posy.insert(0, str(current_inertial_state[1]))
            self.entry_posz.delete(0,tk.END)
            self.entry_posz.insert(0, str(current_inertial_state[2]))
            self.entry_velx.delete(0,tk.END)
            self.entry_velx.insert(0, str(current_inertial_state[3]))
            self.entry_vely.delete(0,tk.END)
            self.entry_vely.insert(0, str(current_inertial_state[4]))
            self.entry_velz.delete(0,tk.END)
            self.entry_velz.insert(0, str(current_inertial_state[5]))
            self.entry_accx.delete(0,tk.END)
            self.entry_accx.insert(0, str(current_inertial_state[6]))
            self.entry_accy.delete(0,tk.END)
            self.entry_accy.insert(0, str(current_inertial_state[7]))
            self.entry_accz.delete(0,tk.END)
            self.entry_accz.insert(0, str(current_inertial_state[8]))
            self.entry_roll.delete(0,tk.END)
            self.entry_roll.insert(0, str(current_inertial_state[9]))
            self.entry_pitch.delete(0,tk.END)
            self.entry_pitch.insert(0, str(current_inertial_state[10]))
            self.entry_yaw.delete(0,tk.END)
            self.entry_yaw.insert(0, str(current_inertial_state[11]))
            self.entry_rollrate.delete(0,tk.END)
            self.entry_rollrate.insert(0, str(current_inertial_state[12]))
            self.entry_pitchrate.delete(0,tk.END)
            self.entry_pitchrate.insert(0, str(current_inertial_state[13]))
            self.entry_yawrate.delete(0,tk.END)
            self.entry_yawrate.insert(0, str(current_inertial_state[14]))
            self.entry_angaccx.delete(0,tk.END)
            self.entry_angaccx.insert(0, str(current_inertial_state[15]))
            self.entry_angaccy.delete(0,tk.END)
            self.entry_angaccy.insert(0, str(current_inertial_state[16]))
            self.entry_angaccz.delete(0,tk.END)
            self.entry_angaccz.insert(0, str(current_inertial_state[17]))








class CarGUI(multiprocessing.Process):
    
    def __init__(self, statePipe, imagePipe, vehicle_names, num_video_feeds, isRGB = True):
        multiprocessing.Process.__init__(self)
        self.statePipe = statePipe
        self.imagePipe = imagePipe
        self.vehicle_names = vehicle_names
        self.num_video_feeds = num_video_feeds
        self.isRGB = isRGB
    
        self.start() # go to the run function
        
    def run(self):
        print("The Application is now running!")
        
        self.root = tk.Tk() # The GUI
        self.root.title("Quadcopter Unreal GUI")
        self.nb = Notebook(self.root)
        
        # Add Main Tab
        self.StateFrame = Frame(self.nb) # Top of Gui 
        self.nb.add(self.StateFrame, text = "Vehicle State")
        
        # Get Notebook Gridded
        self.nb.grid(row = 0, column = 0)
        
        # Configure Video Tab
        self.VideoFrame = Frame(self.nb)
        self.nb.add(self.VideoFrame, text = "Video Feed")
        
        # Configure Plotting Tab
        self.PlottingFrame = Frame(self.nb)
        self.nb.add(self.PlottingFrame, text = "Graphics")
        self.fig = plt.figure(1, figsize= (4,4))
        self.ax = Axes3D(self.fig)
        self.ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.PlottingFrame)
        self.canvas.get_tk_widget().grid(row = 1, column = 0)
        
        # For Switch Vehicle Feeds
        self.current_vehicle_feed = tk.StringVar(self.StateFrame) # Linked to current vehicle choice
        self.switch_vehicle_feed = Combobox(self.StateFrame, textvariable = self.current_vehicle_feed)
        self.switch_vehicle_feed['values'] = self.vehicle_names
        self.switch_vehicle_feed.grid(row = 0, column = 3)
        self.switch_vehicle_feed.current(0)
        
        
        # Labels for state variables
        self.label_posx = Label(self.StateFrame, text = "PosX:")
        self.label_posy = Label(self.StateFrame, text = "PosY:")
        self.label_posz = Label(self.StateFrame, text = "PosZ:")
        self.label_velx = Label(self.StateFrame, text = "Vel X:")
        self.label_vely = Label(self.StateFrame, text = "Vel Y:")
        self.label_velz = Label(self.StateFrame, text = "Vel Z:")
        self.label_accx = Label(self.StateFrame, text = "Acc X:")
        self.label_accy = Label(self.StateFrame, text = "Acc Y:")
        self.label_accz = Label(self.StateFrame, text = "Acc Z:")
#        self.label_roll = Label(self.StateFrame, text = "Roll:")
#        self.label_pitch = Label(self.StateFrame, text = "Pitch:")
#        self.label_yaw = Label(self.StateFrame, text = "Yaw:")
        self.label_rollrate = Label(self.StateFrame, text = "Roll Rate:")
        self.label_pitchrate = Label(self.StateFrame, text = "Pitch Rate:")
        self.label_yawrate = Label(self.StateFrame, text = "Yaw Rate:")
        self.label_angaccx = Label(self.StateFrame, text = "AngAcc X:")
        self.label_angaccy = Label(self.StateFrame, text = "AngAcc Y:")
        self.label_angaccz = Label(self.StateFrame, text = "AngAcc Z:")
        
        
        # Assemble into grid -- No need to pack if you are using grid
        self.label_posx.grid(row = 0, column = 0)
        self.label_posy.grid(row = 1, column = 0)
        self.label_posz.grid(row = 2, column = 0)
        self.label_velx.grid(row = 3, column = 0)
        self.label_vely.grid(row = 4, column = 0)
        self.label_velz.grid(row = 5, column = 0)
        self.label_accx.grid(row = 6, column = 0)
        self.label_accy.grid(row = 7, column = 0)
        self.label_accz.grid(row = 8, column = 0)
#        self.label_roll.grid(row = 9, column = 0)
#        self.label_pitch.grid(row = 10, column = 0)
#        self.label_yaw.grid(row = 11, column = 0)
        self.label_rollrate.grid(row = 9, column = 0)
        self.label_pitchrate.grid(row = 10, column = 0)
        self.label_yawrate.grid(row = 11, column = 0)
        self.label_angaccx.grid(row = 12, column = 0)
        self.label_angaccy.grid(row = 13, column = 0)
        self.label_angaccz.grid(row = 14, column = 0)
        
        # Entries on GUI:
        # Labels for state variables
        self.entry_posx = Entry(self.StateFrame, text = "PosX:")
        self.entry_posy = Entry(self.StateFrame, text = "PosY:")
        self.entry_posz = Entry(self.StateFrame, text = "PosZ:")
        self.entry_velx = Entry(self.StateFrame, text = "Vel X:")
        self.entry_vely = Entry(self.StateFrame, text = "Vel Y:")
        self.entry_velz = Entry(self.StateFrame, text = "Vel Z:")
        self.entry_accx = Entry(self.StateFrame, text = "Acc X:")
        self.entry_accy = Entry(self.StateFrame, text = "Acc Y:")
        self.entry_accz = Entry(self.StateFrame, text = "Acc Z:")
#        self.entry_roll = Entry(self.StateFrame, text = "Roll:")
#        self.entry_pitch = Entry(self.StateFrame, text = "Pitch:")
#        self.entry_yaw = Entry(self.StateFrame, text = "Yaw:")
        self.entry_rollrate = Entry(self.StateFrame, text = "Roll Rate:")
        self.entry_pitchrate = Entry(self.StateFrame, text = "Pitch Rate:")
        self.entry_yawrate = Entry(self.StateFrame, text = "Yaw Rate:")
        self.entry_angaccx = Entry(self.StateFrame, text = "AngAcc X:")
        self.entry_angaccy = Entry(self.StateFrame, text = "AngAcc Y:")
        self.entry_angaccz = Entry(self.StateFrame, text = "AngAcc Z:")

        # Entries in Grid
        self.entry_posx.grid(row = 0, column = 1)
        self.entry_posy.grid(row = 1, column = 1)
        self.entry_posz.grid(row = 2, column = 1)
        self.entry_velx.grid(row = 3, column = 1)
        self.entry_vely.grid(row = 4, column = 1)
        self.entry_velz.grid(row = 5, column = 1)
        self.entry_accx.grid(row = 6, column = 1)
        self.entry_accy.grid(row = 7, column = 1)
        self.entry_accz.grid(row = 8, column = 1)
#        self.entry_roll.grid(row = 9, column = 1)
#        self.entry_pitch.grid(row = 10, column = 1)
#        self.entry_yaw.grid(row = 11, column = 1)
        self.entry_rollrate.grid(row = 9, column = 1)
        self.entry_pitchrate.grid(row = 10, column = 1)
        self.entry_yawrate.grid(row = 11, column = 1)
        self.entry_angaccx.grid(row = 12, column = 1)
        self.entry_angaccy.grid(row = 13, column = 1)
        self.entry_angaccz.grid(row = 14, column = 1)
        
        # The Image Stramer From the simulator:
        self.VideoFeeds = [None for i in range(self.num_video_feeds)]

        
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
    

    def update_image_feed(self):
        scalar = 3
        col_count = 0
        row_count = 0
        if not self.isRGB:
            scalar = 1
        
        while True:
            vehicle_sim_images = self.imagePipe.recv()
            sim_images = vehicle_sim_images[self.current_vehicle_feed.get()]
            for i in range(self.num_video_feeds):
                sim_image = sim_images[:,:,scalar*i:scalar*(i+1)]
                
                if scalar == 1:
                    sim_image = np.reshape(sim_image, (sim_image.shape[0], sim_image.shape[1]))
                if ((i % 3) == 0):
                    col_count = 0
                    row_count += 1
                img = Image.fromarray(sim_image)
                imgtk = ImageTk.PhotoImage(image = img)
                if self.VideoFeeds[i] is None: # Initialize the image panel
                    self.VideoFeeds[i] = Label(self.VideoFrame, image=imgtk)
                    self.VideoFeeds[i].image = imgtk
                    self.VideoFeeds[i].grid(row = row_count, column = col_count)
                else:
                    self.VideoFeeds[i].configure(image = imgtk)
                    self.VideoFeeds[i].image = imgtk
                col_count += 1
            col_count = 0
            row_count = 0
            print("Feed Update")
            
             
    def update_inertial_states(self):
        while True:
            current_inertial_states = self.statePipe.recv()
            current_inertial_state = current_inertial_states[self.current_vehicle_feed.get()]
            print('GUI State Update!')
            self.entry_posx.delete(0,tk.END)
            self.entry_posx.insert(0, str(current_inertial_state[0]))
            self.entry_posy.delete(0,tk.END)
            self.entry_posy.insert(0, str(current_inertial_state[1]))
            self.entry_posz.delete(0,tk.END)
            self.entry_posz.insert(0, str(current_inertial_state[2]))
            self.entry_velx.delete(0,tk.END)
            self.entry_velx.insert(0, str(current_inertial_state[3]))
            self.entry_vely.delete(0,tk.END)
            self.entry_vely.insert(0, str(current_inertial_state[4]))
            self.entry_velz.delete(0,tk.END)
            self.entry_velz.insert(0, str(current_inertial_state[5]))
            self.entry_accx.delete(0,tk.END)
            self.entry_accx.insert(0, str(current_inertial_state[6]))
            self.entry_accy.delete(0,tk.END)
            self.entry_accy.insert(0, str(current_inertial_state[7]))
            self.entry_accz.delete(0,tk.END)
            self.entry_accz.insert(0, str(current_inertial_state[8]))
#            self.entry_roll.delete(0,tk.END)
#            self.entry_roll.insert(0, str(current_inertial_state[9]))
#            self.entry_pitch.delete(0,tk.END)
#            self.entry_pitch.insert(0, str(current_inertial_state[10]))
#            self.entry_yaw.delete(0,tk.END)
#            self.entry_yaw.insert(0, str(current_inertial_state[11]))
            self.entry_rollrate.delete(0,tk.END)
            self.entry_rollrate.insert(0, str(current_inertial_state[9]))
            self.entry_pitchrate.delete(0,tk.END)
            self.entry_pitchrate.insert(0, str(current_inertial_state[10]))
            self.entry_yawrate.delete(0,tk.END)
            self.entry_yawrate.insert(0, str(current_inertial_state[11]))
            self.entry_angaccx.delete(0,tk.END)
            self.entry_angaccx.insert(0, str(current_inertial_state[12]))
            self.entry_angaccy.delete(0,tk.END)
            self.entry_angaccy.insert(0, str(current_inertial_state[13]))
            self.entry_angaccz.delete(0,tk.END)
            self.entry_angaccz.insert(0, str(current_inertial_state[14]))












    