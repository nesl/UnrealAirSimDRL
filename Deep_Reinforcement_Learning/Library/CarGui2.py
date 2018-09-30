# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:33:05 2018

@author: natsn
"""

import tkinter as tk
from tkinter.ttk import Label, Frame, Entry, Notebook, Combobox
import pygame
import pygame.draw
import pygame.time
import threading 
import multiprocessing
import time
import numpy as np
import sys, os
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import  axes3d,Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import cm
from PIL import ImageTk, Image
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\Util\\Virtual_IMU")
from EuclidObjects import Screen, Cube
import SO3Rotation
import ExcelLoader as XL
from ClientAirSimEnvironments.VehicleBase import AirSimImageTypes


class UnrealAirSimGUI(multiprocessing.Process):
    
    def __init__(self, dataPipe, vehicle_names):
        multiprocessing.Process.__init__(self)
        self.dataPipe = dataPipe
        self.vehicle_names = vehicle_names
        self.WIDTH = 580
        self.HEIGHT = 500
        self.update_tracking_interval = 20
        self.start()
    
    def run(self):
        
    # 1/2 Configure the Individual GUI Tabs    
        print("Start GUI Setup!")
        
        self.root = tk.Tk() # The GUI
        self.root.title("Unreal Vehicle GUI")
        self.nb = Notebook(self.root)
        
        # Add Main Tab
        self.StateFrame = Frame(self.nb, width = self.WIDTH, height = self.HEIGHT) # Top of Gui 
        self.nb.add(self.StateFrame, text = "Vehicle State")
        
        # Get Notebook Gridded
        self.nb.grid(row = 0, column = 0)
        
        # Configure Video Tab
        self.VideoFrame = Frame(self.nb, width = self.WIDTH, height = self.HEIGHT)
        self.nb.add(self.VideoFrame, text = "Video Feed")
        
        # Configure Plotting Tab
        self.PlottingFrame = Frame(self.nb, width = self.WIDTH, height = self.HEIGHT)
        self.nb.add(self.PlottingFrame, text = "Track n' Map")
        self.fig = plt.figure(1, figsize= (4,4))
        self.ax = Axes3D(self.fig)
        self.last_3d_viz_x_pos = 0
        self.last_3d_viz_y_pos = 0
        self.last_3d_viz_z_pos = 0
        self.ax.scatter(self.last_3d_viz_x_pos,self.last_3d_viz_y_pos,self.last_3d_viz_z_pos)
        self.canvas = FigureCanvasTkAgg(self.fig, self.PlottingFrame)
        self.canvas.get_tk_widget().grid(row = 1, column = 0)
        
        # Configure Virtual IMU Tab:
        self.embed = tk.Frame(self.root, width = self.WIDTH, height = self.HEIGHT)
        self.nb.add(self.embed, text="Virtual IMU")
        os.environ['SDL_WINDOWID'] = str(self.embed.winfo_id())  #Tell pygame's SDL window which window ID to use
        
        # Start PYGAME for IMU Visualization
        pygame.init()
        self.screen = Screen(500,500, scale=1.5)
        self.cube = Cube(40,40,40)
        # END Initialization
        self.root.update()
        
    # 2/2 Configure the Labels and Entries on the GUI    
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
        self.label_rollrate = Label(self.StateFrame, text = "Roll Rate:")
        self.label_pitchrate = Label(self.StateFrame, text = "Pitch Rate:")
        self.label_yawrate = Label(self.StateFrame, text = "Yaw Rate:")
        self.label_angaccx = Label(self.StateFrame, text = "AngAcc X:")
        self.label_angaccy = Label(self.StateFrame, text = "AngAcc Y:")
        self.label_angaccz = Label(self.StateFrame, text = "AngAcc Z:")
        self.label_roll = Label(self.StateFrame, text = "Roll:")
        self.label_pitch = Label(self.StateFrame, text = "Pitch:")
        self.label_yaw = Label(self.StateFrame, text = "Yaw:")        
        
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
        self.label_rollrate.grid(row = 9, column = 0)
        self.label_pitchrate.grid(row = 10, column = 0)
        self.label_yawrate.grid(row = 11, column = 0)
        self.label_angaccx.grid(row = 12, column = 0)
        self.label_angaccy.grid(row = 13, column = 0)
        self.label_angaccz.grid(row = 14, column = 0)
        self.label_roll.grid(row = 15, column = 0)
        self.label_pitch.grid(row = 16, column = 0)
        self.label_yaw.grid(row = 17, column = 0)
        
        # Entries for State Updates:
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

        # Entries Gridded
        self.entry_posx.grid(row = 0, column = 1)
        self.entry_posy.grid(row = 1, column = 1)
        self.entry_posz.grid(row = 2, column = 1)
        self.entry_velx.grid(row = 3, column = 1)
        self.entry_vely.grid(row = 4, column = 1)
        self.entry_velz.grid(row = 5, column = 1)
        self.entry_accx.grid(row = 6, column = 1)
        self.entry_accy.grid(row = 7, column = 1)
        self.entry_accz.grid(row = 8, column = 1)
        self.entry_roll.grid(row = 15, column = 1)
        self.entry_pitch.grid(row = 16, column = 1)
        self.entry_yaw.grid(row = 17, column = 1)
        self.entry_rollrate.grid(row = 9, column = 1)
        self.entry_pitchrate.grid(row = 10, column = 1)
        self.entry_yawrate.grid(row = 11, column = 1)
        self.entry_angaccx.grid(row = 12, column = 1)
        self.entry_angaccy.grid(row = 13, column = 1)
        self.entry_angaccz.grid(row = 14, column = 1)
        
        # Meta Data For the State Page
        self.entry_action = Entry(self.StateFrame, text = "Action")
        self.entry_action_name = Entry(self.StateFrame, text = "Action Name")
        self.entry_env_state = Entry(self.StateFrame, text = "Env State")
        self.entry_mode = Entry(self.StateFrame, text = "GUI Mode")
        self.entry_act_time = Entry(self.StateFrame, text = "Action Time")
        self.entry_sim_image_time = Entry(self.StateFrame, text = "Sim Image Get Time")
        self.entry_sim_state_time = Entry(self.StateFrame, text = "Sim State Get Time")
        self.entry_reward_time = Entry(self.StateFrame, text = "Sim Calc Reward Time")
        self.entry_step_time = Entry(self.StateFrame, text = "Step Time")
        self.entry_reward = Entry(self.StateFrame, text = "Reward")
        self.entry_done = Entry(self.StateFrame, text = "Done Flag")
        
        
        self.label_action = Label(self.StateFrame, text = "Action:")
        self.label_action_name = Label(self.StateFrame, text = "Action Name:")
        self.label_env_state = Label(self.StateFrame, text = "Env State:")
        self.label_mode = Label(self.StateFrame, text = "GUI Mode:")
        self.label_act_time = Label(self.StateFrame, text = "Action Time:")
        self.label_sim_image_time = Label(self.StateFrame, text = "Sim Image Get Time:")
        self.label_sim_state_time = Label(self.StateFrame, text = "Sim State Get Time:")
        self.label_reward_time = Label(self.StateFrame, text = "Calc Reward Time:")
        self.label_step_time = Label(self.StateFrame, text = "Env Step Time:")
        self.label_reward = Label(self.StateFrame, text = "Reward:")
        self.label_done = Label(self.StateFrame, text = "Done:")
        
        # Grid Meta Data Display
        self.label_action.grid(row = 5, column = 2)
        self.label_action_name.grid(row = 6, column = 2)
        self.label_env_state.grid(row = 7, column = 2)
        self.label_mode.grid(row = 8, column = 2)
        self.label_act_time.grid(row = 9, column = 2)
        self.label_sim_image_time.grid(row = 10, column = 2)
        self.label_sim_state_time.grid(row = 11, column = 2)
        self.label_reward_time.grid(row = 12, column = 2)
        self.label_step_time.grid(row = 13, column = 2)
        self.label_reward.grid(row = 14, column = 2)
        self.label_done.grid(row = 15, column = 2)
        
        self.entry_action.grid(row = 5, column = 3)
        self.entry_action_name.grid(row = 6, column = 3)
        self.entry_env_state.grid(row = 7, column = 3)
        self.entry_mode.grid(row = 8, column = 3)
        self.entry_act_time.grid(row = 9, column = 3)
        self.entry_sim_image_time.grid(row = 10, column = 3)
        self.entry_sim_state_time.grid(row = 11, column = 3)
        self.entry_reward_time.grid(row = 12, column = 3)
        self.entry_step_time.grid(row = 13, column = 3)
        self.entry_reward.grid(row = 14, column = 3)
        self.entry_done.grid(row = 15, column = 3)
        
        # Initialize the Vehicle's Virtual IMU Visualization
        self.label_yaw = Label(self.embed, text = "Yaw:")
        self.label_pitch = Label(self.embed, text = "Pitch:")
        self.label_roll = Label(self.embed, text = "Roll:")
        self.label_yaw.place(x=500,y = 0) # Place the Labels on the far right of the frame
        self.label_pitch.place(x=500, y = 50)
        self.label_roll.place(x=500, y = 100)
        
        self.entry_imu_yaw = Entry(self.embed, text = "Yaw:")
        self.entry_imu_pitch = Entry(self.embed, text = "Pitch:")
        self.entry_imu_roll = Entry(self.embed, text = "Roll:")
        self.entry_yaw.place(x = 500, y = 25)
        self.entry_pitch.place(x = 500, y = 75)
        self.entry_roll.place(x = 500, y = 125)
        
        print("GUI Setup DONE!")
        t_upd = threading.Thread(target = self.updates)
        t_upd.start()
        self.root.mainloop()
        

    
    def updates(self):
        time.sleep(1.5)
        while True:
            # Data comes in as a dictionary of 'obs', 'state', 'meta'
            data = self.dataPipe.recv()
            vehicle_name = self.current_vehicle_feed.get()
            # Run the inertial update thread
            self.t_states = threading.Thread(target = self.update_inertial_states, 
                                             args = (data[vehicle_name]['state'],))
            self.t_states.start() # Start Updater thread by setting the event
            # Run the Image Data thread
            self.t_imgs = threading.Thread(target = self.update_image_feeds,
                                           args = (data[vehicle_name]['obs'],))
            self.t_imgs.start()
            # Run the meta data update
            self.t_meta = threading.Thread(target = self.update_metas,
                                            args = (data[vehicle_name]['meta'],))
            self.t_meta.start()
            
            self.t_v_imu_viz = threading.Thread(target = self.update_virtual_imu_visualization,
                                                args = (data[vehicle_name]['state'],))
            self.t_v_imu_viz.start()
            
            self.t_3d_track_viz = threading.Thread(target = self.update_object_3DVisualization, 
                                                   args = (data[vehicle_name]['state'],))
            self.t_3d_track_viz.start()
            
            # Join Threads
            self.t_states.join()
            self.t_imgs.join()
            self.t_meta.join()
            self.t_v_imu_viz.join()
            self.t_3d_track_viz.join()
            
        
    def update_metas(self, data):
        meta = data
        self.entry_action_name.delete(0,tk.END)
        self.entry_action_name.insert(0,str(meta['action_name']))
        self.entry_action.delete(0,tk.END)
        self.entry_action.insert(0,str(meta['action']))
        self.entry_env_state.delete(0,tk.END)
        self.entry_env_state.insert(0,str(meta['env_state']))
        self.entry_mode.delete(0,tk.END)
        self.entry_mode.insert(0,str(meta['mode']))
        self.entry_act_time.delete(0, tk.END)
        self.entry_act_time.insert(0, str(meta['times']['act_time']))
        self.entry_sim_image_time.delete(0,tk.END)
        self.entry_sim_image_time.insert(0,str(meta['times']['sim_img_time']))
        self.entry_sim_state_time.delete(0,tk.END)
        self.entry_sim_state_time.insert(0,str(meta['times']['sim_state_time']))
        self.entry_reward_time.delete(0,tk.END)
        self.entry_reward_time.insert(0,str(meta['times']['reward_time']))
        self.entry_step_time.delete(0, tk.END)
        self.entry_step_time.insert(0, str(meta['times']['step_time']))
        self.entry_reward.delete(0, tk.END)
        self.entry_reward.insert(0, str(meta['reward']))
        self.entry_done.delete(0, tk.END)
        self.entry_done.insert(0, str(meta['done']))

    def update_image_feeds(self, data):
        scalar = 3
        col_count = 0
        row_count = 0
        
        sim_images = data
        print("GUI Image Update:")
        start_time = time.time()
        
        for key in sim_images:
            if len(sim_images[key][0]) > 0:
                
        
        
        for i in range(self.num_video_feeds):
            sim_image = sim_images[:,:,scalar*i:scalar*(i+1)]
            if scalar == 1:
                sim_image = np.reshape(sim_image, (sim_image.shape[0], sim_image.shape[1]))
            if ((i % 3) == 0):
                col_count = 0
                row_count += 1
            #print('sim image shape ', sim_image.shape, type(sim_image), sim_image, self.isNormal)
            if self.isNormal:
                sim_image = np.array(sim_image * 255, dtype = np.uint8)
            else:
                sim_image = np.array(sim_image, dtype = np.uint8)
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
        print("Feed Update Time: ", time.time() - start_time)
    

    
    def update_inertial_states(self, data):
            #print(current_inertial_states)
            current_inertial_state = data
            quatx, quaty, quatz, quatw = current_inertial_state[15:]
            yaw, pitch, roll = SO3Rotation.quaternion_to_euler_angles((quatw, quatx, quaty, quatz))

            print('GUI State Update!')
            start_time = time.time()
            #print(current_inertial_states, current_inertial_state)
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
            self.entry_roll.insert(0, str(roll))
            self.entry_pitch.delete(0,tk.END)
            self.entry_pitch.insert(0, str(pitch))
            self.entry_yaw.delete(0,tk.END)
            self.entry_yaw.insert(0, str(yaw))
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
            print('GUI State Update Time! ', time.time() - start_time)        
             

    def update_virtual_imu_visualization(self, data):

        quatx = data[15]
        quaty = data[16]
        quatz = data[17]
        quatw = data[18]
        yaw, pitch, roll = SO3Rotation.quaternion_to_euler_angles((quatw, quatx, quaty, quatz))
        self.entry_yaw.delete(0,tk.END)
        self.entry_yaw.insert(0, str(yaw))
        self.entry_pitch.delete(0,tk.END)
        self.entry_pitch.insert(0, str(pitch))
        self.entry_roll.delete(0,tk.END)
        self.entry_roll.insert(0, str(roll))
        q = Quaternion(quatw, quatx, quaty, quatz).normalized()
        self.cube.draw(self.screen,q)
        event = pygame.event.poll()
        pygame.display.flip()
        pygame.time.delay(10) # ms
        self.cube.erase(self.screen)
        #TKINTER
        self.root.update()

    def update_object_3DVisualization(self, data):
        if self.update_tracking_interval % 20 == 0:
            print("RUNNING 3D VISULIZATION")
            print(data[0], data[1], data[2])
            xpos = data[0]
            ypos = data[1]
            zpos = data[2]
            self.ax.plot3D([self.last_3d_viz_x_pos, xpos],
                           [self.last_3d_viz_y_pos,ypos],
                           zs = [self.last_3d_viz_z_pos, zpos])
            self.ax.scatter(xpos, ypos, zpos, color = 'green')
            self.canvas.draw()
            
            self.last_3d_viz_x_pos = xpos
            self.last_3d_viz_y_pos = ypos
            self.last_3d_viz_z_pos = zpos
        self.update_tracking_interval += 1
        
        
        
        
        
        
        
        
       