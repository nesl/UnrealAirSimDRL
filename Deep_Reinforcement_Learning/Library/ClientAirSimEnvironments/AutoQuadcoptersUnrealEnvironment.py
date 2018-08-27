# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:51:44 2018

@author: natsn
"""

import numpy as np
import time
from airsim import client
from airsim.types import Vector3r, Quaternionr
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
import AirSimGUI
import multiprocessing 
import threading
from ImageProcessing import trim_append_state_vector, fill_state_vector




# The AirSim Environment Class for Quadrotor Safety
class MultiQuadcoptersUnrealEnvironment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    def __init__(self, vehicle_names, max_altitude = 18,
                 min_altitude = 2,
                 image_mask_FC_FR_FL = [True, False, False], # Front Center, Front right, front left
                 reward_function = drone_forest_racer_rewarding_function,
                 mode = "both_rgb"):
        self.vehicle_names = vehicle_names
        self.reward_function = reward_function
        self.mode = mode
        
        #self.ImageUpdateProcess = multiprocessing.Pool(processes = 3)
        #self.parent_conn, self.child_conn = multiprocessing.Pipe()
        #self.lock = threading.Lock()
        
        self.scaling_factor = .200 # Used as a constant gain factor for the action throttle. 
        self.action_duration = .150 # Each action lasts this many seconds
        
        # Gains on the control commands sent to the quadcopter #
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 18 # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd, Rdd, Pdd, Ydd
        self.count_drone_actions = 14 # 6 Linear, 6 angular, 1 hover, 1 No Op (Dont change anything)
        
        # Initialize the current inertial state
        self.current_inertial_states = dict.fromkeys(self.vehicle_names, np.array(np.zeros(self.count_inertial_state_variables)))
        
        # Initialize the IMAGE variables -- We Take in Front Center, Right, Left
        self.images_rgb = dict.fromkeys(self.vehicle_names, None)
        self.images_rgba = dict.fromkeys(self.vehicle_names, None)
        
        self.image_mask_rgb = np.array([ [0+3*i,1+3*i,2+3*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_rgba = np.array([ [0+4*i,1+4*i,2+4*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_FC_FR_FL = image_mask_FC_FR_FL
        
        self.initial_position = dict.fromkeys(self.vehicle_names, None)
        self.initial_velocity = dict.fromkeys(self.vehicle_names, None)
        self.extra_metadata = dict.fromkeys(self.vehicle_names, None)
        # Set max altitude the quadcopter can hover
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        
        # Connect to the AirSim simulator and begin:
        print('Initializing Client')
        self.client = client.MultirotorClient()
        self.client.confirmConnection()
        for name in self.vehicle_names:
            self.client.enableApiControl(True, name)
            self.client.armDisarm(True, name)
        time.sleep(.5)
        print('Initialization Complete!')
        
        # Timing Operations Initialize
        self.dt = 0
        self.tic = 0
        self.toc = 0
        

    # The List of all possible actions for the Quadcopter are as follows:
    # 0: No Action
    # 1: Hover 
    # 2: Move by Velocity in X
    # 3: Move by Velocity in Y
    # 4: Move by Velocity in Z
    # 5: Move by Velocity in -X
    # 6: Move by Velocity in -Y
    # 7: Move by Velocity in -Z
    # 8: Move by Angle +Roll
    # 9: Move by Angle +Pitch
    # 10: Move by Angle +Yaw
    # 11: Move by Angle -Roll
    # 12: Move by Angle -Pitch
    # 13: Move by Angle -Yaw
    
    # Action is an integer value corresponding to above
    # Three step modes: 
    # 1: Inertial: Returns the quadcopter's inertial state information from the simulator
    # 2: Image: Returns the quadcopter's image information from the simulator
    def steps(self, actions):
        
        # 1. Take action in the simulator based on the agents action choice, 1 for each vehicle
        self.do_actions(actions)
        
        # 2: Update the environment state variables after the action takes place for each vehicle
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        rewards, dones = self.calc_rewards()
        
        if self.mode == "inertial":
            return (self.current_inertial_states, rewards, dones, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.get_rgbs(), rewards, dones, self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(), rewards, dones, self.extra_metadata)
        elif self.mode == "rgba":
            return (self.get_rgbas(), rewards, dones, self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), rewards, dones, self.extra_metadata)
        elif self.mode == "gray_normal":
            return (self.rgbs2grays_normal(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_statse, self.get_rgbs(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2rgbs_normal(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays_normal(), rewards, dones, self.extra_metadata)
        else:
            print("invalid Mode!")
    
    def get_last_obs(self, mode = None):
        if mode is None:
            mode = self.mode
        
        if mode == "inertial":
            return self.current_inertial_states
        elif mode == "rgb":
            return self.get_rgbs()
        elif mode == "rgb_normal":
            return self.rgbs2rgbs_normal()
        elif mode == "rgba":
            return self.get_rgbas()
        elif mode == "gray":
            return self.rgbs2grays()
        elif mode == "gray_normalized": # 0 - 1
            return self.rgbs2grays_normal()
        elif mode == "both_rgb":
            return (self.current_inertial_states, self.get_rgbs())
        elif mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2srgb_normal())
        elif mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas())
        elif mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays())
        elif mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays_normal())
        else:
            print("invalid Mode!")
    
    def rgbs2rgbs_normal(self):
        vehicle_rgbs = dict.fromkeys(self.vehicle_names)
        rgbs_norm = []
        num_imgs = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int))
        for vn in self.vehicle_names:
            for i in range(num_imgs):
                rgbs_norm.append(self.rgb2rgb_normal(self.images_rgb[vn][:,:,self.image_mask_rgb[3*i:3*(i+1)]]))
            rgbs_normal_cube = rgbs_norm[0]
            for i in range(num_imgs - 1):
                rgbs_normal_cube = np.dstack((rgbs_normal_cube, rgbs_norm[i+1]))
            vehicle_rgbs[vn] = rgbs_normal_cube
        return vehicle_rgbs
    
    def rgb2rgb_normal(self, rgb):
        # Special function to turn the RGB cube into a gray scale cube:
        return np.array(rgb - np.atleast_3d(np.mean(rgb, axis = 2)), dtype = np.float32) / np.atleast_3d(np.std(rgb, axis = 2) + .001)
    
    # Returns all grays from rgb, given your environment settings
    # Works on the internal 2 dim stacked vision cube images_rgb/a
    def rgbs2grays(self):
        vehicle_grays = dict.fromkeys(self.vehicle_names)
        grays = []
        num_imgs = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int))
        for vn in self.vehicle_names:
            for i in range(num_imgs):
                grays.append(self.rgb2gray(self.images_rgb[vn][:,:,self.image_mask_rgb[3*i:3*(i+1)]], isGray3D = True))
            graycube = grays[0]
            for i in range(num_imgs - 1):
                graycube = np.dstack((graycube, grays[i+1]))
            vehicle_grays[vn] = graycube
        return vehicle_grays
    
    def rgbs2grays_normal(self):
        vehicle_grays = dict.fromkeys(self.vehicle_names)
        grays = []
        num_imgs = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int))
        for vn in self.vehicle_names:
            for i in range(num_imgs):
                grays.append(np.array(self.rgb2gray(self.images_rgb[vn][:,:,self.image_mask_rgb[3*i:3*(i+1)]], isGray3D = True), dtype = np.float32) / 255)
            graycube = grays[0]
            for i in range(num_imgs - 1):
                graycube = np.dstack((graycube, grays[i+1]))
            vehicle_grays[vn] = graycube
        return vehicle_grays
    
    def rgb2gray(self, rgb, isGray3D = True): # puts a 1 in channel 2 : (0,1,2=>1)
        # Special function to turn the RGB cube into a gray scale cube:
        if isGray3D:
            return np.array(np.atleast_3d(np.mean(rgb, axis = 2)), dtype = np.uint8)
        else:
            return np.array(np.mean(rgb, axis = 2), dtype = np.uint8)
    def get_rgbs(self):
        rgbs = dict.fromkeys(self.vehicle_names, None)
        for vn in self.vehicle_names:
            rgbs[vn] = self.images_rgb[vn][:,:,self.image_mask_rgb]
        return rgbs
    def get_rgbas(self):
        rgbas = dict.fromkeys(self.vehicle_names, None)
        for vn in self.vehicle_names:
            rgbas[vn] = self.images_rgba[vn][:,:,self.image_mask_rgba]
        return rgbas
    
    def get_new_images(self, vehicleName):
        # Wait to grad images
        tic = time.time()
        print("Collecting Images from AirSim")
        if (self.image_mask_FC_FR_FL[0] and  self.image_mask_FC_FR_FL[1] and  self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([
            client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
            client.ImageRequest("1", client.ImageType.Scene, False, False), # Front Right
            client.ImageRequest("2", client.ImageType.Scene, False, False)], vehicle_name = vehicleName) # Front Left
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            
            img1d_FR = np.fromstring(images[1].image_data_uint8, dtype=np.uint8) 
            img_rgba_FR = np.array(img1d_FR.reshape(images[1].height, images[1].width, 4), dtype = np.uint8)
            img_rgb_FR = img_rgba_FR[:,:,0:3]
            
            img1d_FL = np.fromstring(images[2].image_data_uint8, dtype=np.uint8) 
            img_rgba_FL = np.array(img1d_FL.reshape(images[2].height, images[2].width, 4), dtype = np.uint8)
            img_rgb_FL = img_rgba_FL[:,:,0:3]
            
            # Can either use the RGBA images or the RGB Images
            
            self.images_rgba[vehicleName] = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
            self.images_rgb[vehicleName] = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
            print('Images Collected!')
            print("Time to Grab Images: ", time.time() - tic)
             
        # We Just want front        
        elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)], vehicle_name = vehicleName)
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            self.images_rgba[vehicleName] = img_rgba_FC
            self.images_rgb[vehicleName] = img_rgb_FC
            print('Images Collected!')
            print("Time to Grab Images: ", time.time() - tic)
        
        else:
            print("A screw up in set new images")

    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        self.client.simPause(True)
        # Get Base Inertial States
        for vn in self.vehicle_names:
            state = self.client.simGetGroundTruthKinematics(vehicle_name = vn)
            pos = (state['position']['x_val'],state['position']['y_val'],-1*state['position']['z_val'])
            vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'], -1*state['linear_velocity']['z_val'])
            acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
            orien = (state['orientation']['x_val'],state['orientation']['y_val'],state['orientation']['z_val'])
            angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
            angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
            
            # Collect and Display Elapsed Time Between States
            self.toc = time.time()
            self.dt = self.toc - self.tic
            
            # Store the current state
            self.current_inertial_states[vn] = np.array([pos[0],
                                                    pos[1],
                                                    pos[2],
                                                    vel[0],
                                                    vel[1],
                                                    vel[2],
                                                    acc[0], acc[1], acc[2],
                                                    orien[0], orien[1], orien[2], 
                                                    angVel[0], angVel[1], angVel[2],
                                                    angAcc[0], angAcc[1], angAcc[2]])
            # Load new images into env from sim
            self.get_new_images(vn)
            # Store X and Y position information as well
            self.extra_metadata[vn] = self.dt
            # Start the timer again as the next state is saved. Return the current state and meta-data 
            self.tic = time.time()
        self.client.simPause(False)
    #Ask the simulator to perform control command or to block control command
    def do_actions(self, actions):
        action_list = []
        quadStates = []
        # Either move by velocity, or move by angle at each timestep
        assert len(actions) == len(self.vehicle_names)
        for vn in self.vehicle_names:
            action_list.append(self.get_RC_action(act_num = actions[vn]))
            quadStates.append(self.client.simGetGroundTruthKinematics(vehicle_name = vn))
        for i in range(len(self.vehicle_names)):
            if action_list[i][0] is not None: #  Move By Velocity
                quad_vel = quadStates[i]['linear_velocity']
                self.client.moveByVelocityAsync(quad_vel['x_val'] + action_list[i][0][0], 
                                                quad_vel['y_val'] + action_list[i][0][1], 
                                                quad_vel['z_val'] + action_list[i][0][2], 
                                                self.action_duration, vehicle_name = self.vehicle_names[i])
            elif action_list[i][1] is not None: # Move By Angle
                quad_pry = quadStates[i]['orientation']
                self.client.moveByAngleZAsync(quad_pry['y_val'] + action_list[i][1][0], 
                                              quad_pry['x_val'] + action_list[i][1][1], 
                                              quad_pry['z_val'], quad_pry['w_val'] + action_list[i][1][2], 
                                              self.action_duration, vehicle_name = self.vehicle_names[i])
            elif action_list[i][2] is not None: # Move By Hover
                self.client.hoverAsync(vehicle_name = self.vehicle_names[i])
            else:
                print("error in do action")
        time.sleep(self.action_duration)
    # The reward function is broken into an intervention signal and non intervention signal, as the actions have different
        # state dependencies
    def calc_rewards(self):
        rewards = dict.fromkeys(self.vehicle_names)
        dones = dict.fromkeys(self.vehicle_names)
        for vn in self.vehicle_names:
            collision_info = self.client.simGetCollisionInfo(vehicle_name = vn)
            r, d = self.reward_function(collision_info, self.current_inertial_states[vn], self.max_altitude, self.min_altitude)
            rewards[vn] = r
            dones[vn] = d
        return (rewards, dones)
    
    # Use this function to select a random movement or
    # if act_num is set to one of the functions action numbers, then all information about THAT action is returned
    # if act_num is not set, then the function will generate a random action
    def get_RC_action(self, act_num = None):
            
            # Random drone action to be selected, if none are specified
            rand_act_n = 0
            # If action number is set, return the corresponding action comand
            if act_num is not None:
                rand_act_n = act_num
            # Else, use a random action
            else:
                rand_act_n = np.random.randint(0,self.count_drone_actions)
                
            move_by_vel = None
            move_by_angle = None
            move_by_hover = None
            
            if rand_act_n == 0: # no action
                move_by_vel = (0, 0, 0)
            elif rand_act_n == 1:
                move_by_hover = True
            elif rand_act_n == 2:
                move_by_vel = (self.scaling_factor, 0, 0)
            elif rand_act_n == 3:
                move_by_vel = (0, self.scaling_factor, 0)
            elif rand_act_n == 4:
                move_by_vel = (0, 0, self.scaling_factor)
            elif rand_act_n == 5:
                move_by_vel = (-self.scaling_factor, 0, 0)
            elif rand_act_n == 6:
                move_by_vel = (0, -self.scaling_factor, 0)
            elif rand_act_n == 7:
                move_by_vel = (0, 0, -self.scaling_factor)
            elif rand_act_n == 8:
                move_by_angle = (self.scaling_factor, 0, 0)
            elif rand_act_n == 9:
                move_by_angle = (0, self.scaling_factor, 0)
            elif rand_act_n == 10:
                move_by_angle = (0, 0, self.scaling_factor)
            elif rand_act_n == 11:
                move_by_angle = (-self.scaling_factor, 0, 0)
            elif rand_act_n == 12:
                move_by_angle = (0, -self.scaling_factor, 0)
            elif rand_act_n == 13:
                move_by_angle = (0, 0, -self.scaling_factor)
                
            # Move by angle or move by velocity (one of the two will be set to None), meaning we either move by one or the other to generate a trajectory
            return move_by_vel, move_by_angle, move_by_hover, rand_act_n

    def action_num(self, actionName):
        dic = {0: 'No Action', 1: 'Hover', 2: 'Vx',3: 'Vy',
               4: 'Vz', 5: '-Vx', 6: '-Vy', 7: '-Vz',
               8: '+Roll', 9: '+Pitch', 10: '+Yaw', 11: '-Roll',
               12: '-Pitch', 13: '-Yaw'}
        return dic[actionName]
    def action_name(self, actionNum):
        dic = {'No Action': 0, 'Hover': 1, 'Vx': 2,'Vy' : 3,
               'Vz': 4, '-Vx': 5, '-Vy': 6 , '-Vz' : 7,
               '+Roll' : 8,'+Pitch' : 9, '+Yaw': 10, '-Roll': 11,
               '-Pitch' : 12, '-Yaw': 13}
        return dic[actionNum]
    
    def reset(self):
        # Reset the Copters
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        for vn in self.vehicle_names:
            self.client.enableApiControl(True, vn)

        # Quickly raise the quadcopters to a few meters -- Speed up
        for i in range(12):
            for vn in self.vehicle_names:
                state = self.client.simGetGroundTruthKinematics(vehicle_name = vn)
                quad_vel = state['linear_velocity']
                quad_offset = (0, 0, -self.scaling_factor)
                self.client.moveByVelocityAsync(quad_vel['x_val']+quad_offset[0], 
                                                quad_vel['y_val']+quad_offset[1], 
                                                quad_vel['z_val'] + quad_offset[2], 
                                                self.action_duration, vehicle_name = vn)
            time.sleep(self.action_duration)
        
        for vn in self.vehicle_names:
            state = self.client.simGetGroundTruthKinematics(vehicle_name = vn)
            self.initial_position[vn] = (state['position']['x_val'],
                   state['position']['y_val'],
                   state['position']['z_val']*-1)
            
            self.initial_velocity[vn] = (state['linear_velocity']['x_val'],
                   state['linear_velocity']['y_val'],
                   state['linear_velocity']['z_val'])
            
        print("Initial Quad Position: ", self.initial_position[self.vehicle_names[0]])
        print('Reset Complete')
        
        # Start Timer for episode's first step:
        self.dt = 0
        self.tic = time.time()
        time.sleep(.5)
        
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        if self.mode == "inertial":
            return (self.current_inertial_states, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.get_rgbs(), self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "rgba":
            return (self.get_rgbas(), self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "gray_normal":
            return (self.rgbs2grays_normal(), self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_states, self.get_rgbs(), self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas(), self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays_normal(), self.extra_metadata)
        else:
            print("invalid Mode!")
 





