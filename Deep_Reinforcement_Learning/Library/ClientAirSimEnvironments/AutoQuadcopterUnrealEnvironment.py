# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:50:26 2018

@author: natsn
"""


import numpy as np
import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")
from airsim import client
from airsim.types import Vector3r
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")
import AirSimGUI
import multiprocessing 
from ImageProcessing import trim_append_state_vector
from RewardingFunctions import drone_forest_racer_rewarding_function



# The AirSim Environment Class for Quadrotor Safety
class AutoQuadcopterUnrealEnvironment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    # If you stack your image frames

    def __init__(self, vehicle_name = "Drone1",
                 max_altitude = 12,
                 min_altitude = .45,
                 time_to_exec_hover = 1,
                 image_mask_FC_FR_FL = [True, True, True], # Front Center, Front right, front left
                 sim_mode = "both_rgb",
                 IMG_HEIGHT = 128,
                 IMG_WIDTH = 128,
                 IMG_STEP = 3,
                 reward_function = drone_forest_racer_rewarding_function):
        
        self.reward_function = reward_function
        self.mode = sim_mode
        self.time_to_exec_hover = time_to_exec_hover
        
        self.scaling_factor = .30 # Used as a constant gain factor for the action throttle. 
        self.action_duration = .10 # (ms) Each action lasts this many seconds
        
            
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 18 # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd, Rdd, Pdd, Ydd
        self.count_drone_actions = 14 # 6 Linear, 6 angular, 1 hover, 1 No Op (Dont change anything)
        
        # Simulator Image setup
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        isRGB = False
        IMG_CHANNELS = 1
        if 'rgb' in self.mode:
            isRGB = True
            IMG_CHANNELS = 3
        isNormal = False
        if 'normal' in self.mode:
            isNormal = True
        self.IMG_CHANNELS = IMG_CHANNELS
        self.IMG_STEP = IMG_STEP
        self.IMG_VIEWS = np.sum(np.array(image_mask_FC_FR_FL, dtype = np.int))
        # Initialize the container that holds the sequence of images from the simulator
        self.obs4 = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS * self.IMG_STEP * self.IMG_VIEWS))
        
        # Initialize the current inertial state
        self.current_inertial_state = np.array(np.zeros(self.count_inertial_state_variables))
        
        # Initialize the IMAGE variables -- We Take in Front Center, Right, Left
        self.images_rgb = None
        self.images_rgba = None
        self.image_mask_rgb = np.array([ [0+3*i,1+3*i,2+3*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_rgba = np.array([ [0+4*i,1+4*i,2+4*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_FC_FR_FL = image_mask_FC_FR_FL
        
        # Set max altitude the quadcopter can hover
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        
        # Connect to the AirSim simulator and begin:
        print('Initializing Client')
        self.client = client.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print('Initialization Complete!')
        print("Setting Camera Views")
        orien = Vector3r(0, 0, 0)
        self.client.simSetCameraOrientation(0, orien) #radians
        orien = Vector3r(0, .12, -np.pi/9)
        self.client.simSetCameraOrientation(1, orien)
        orien = Vector3r(0, .12, np.pi/9)
        self.client.simSetCameraOrientation(2, orien)
        # Reset Collion Flags
        print("Setting Camera Views DONE!")
        
        # Set up GUI Video Feeder
        self.gui_data = {'obs': None, 'state': None, 'meta': None}
        self.vehicle_name = vehicle_name
        num_video_feeds = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int)) * IMG_STEP
        GUIConn, self.simEnvDataConn = multiprocessing.Pipe()
        self.app = AirSimGUI.QuadcopterGUI(GUIConn, vehicle_names = [vehicle_name],
                                                   num_video_feeds = num_video_feeds, isRGB = isRGB, isNormal = isNormal)
        
        # Timing Operations Initialize
        self.time_to_do_action = 0
        self.time_to_grab_images = 0
        self.time_to_grab_states = 0
        self.time_to_calc_reward = 0
        self.time_to_step = 0
        self.extra_metadata = None
        

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
    # Note: RGBA doesnt do much but will be fixed when depth for depth sensing is added
    def step(self, action):
        
        tic = time.time()
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        reward, done = self.calc_reward()
        
        self.time_to_step = time.time() - tic

        # Send the data collected off to gui
        self.send_to_gui(action, reward, done)
        
        if self.mode == "inertial":
            return (self.current_inertial_state, reward, done, self.extra_metadata)
        elif self.mode == "rgb":
            # Reduce dimentionality of obs
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgb[:,:,self.image_mask_rgb], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "rgb_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "rgba":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "gray":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "gray_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays() / 255, pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "both_rgb":
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgb[:,:, self.image_mask_rgb], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "both_rgba": 
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgba[:,:, self.image_mask_rgba], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "both_gray":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, reward, done, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays()/ 255, pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, reward, done, self.extra_metadata)
        else:
            print("invalid Mode!")
    

    def send_to_gui(self, action, reward, done):
        print(action)
        self.extra_metadata = {'action': action, 'action_name': self.action_name(action), 'env_state': {'resetting': False, 'running': True},
                               'mode': self.mode, 'reward': reward, 'done': done, 'times': {'act_time': self.time_to_do_action,
                                                            'sim_img_time': self.time_to_grab_images,
                                                            'sim_state_time': self.time_to_grab_states,
                                                            'reward_time': self.time_to_calc_reward,
                                                            'step_time': self.time_to_step}}
        
        self.gui_data['state'] = self.current_inertial_state
        self.gui_data['obs'] = self.obs4
        self.gui_data['meta'] = self.extra_metadata
        self.simEnvDataConn.send({self.vehicle_name : self.gui_data})
        
        
    
    def get_last_obs(self, mode = None):
        if mode is None:
            mode = self.mode
        
        if mode == "inertial":
            return self.current_inertial_state
        elif mode == "rgb":
            return self.images_rgb[:,:,self.image_mask_rgb]
        elif mode == "rgb_normal":
            return self.rgbs2rgbs_normal()
        elif mode == "rgba":
            return self.images_rgba[:,:,self.image_mask_rgba]
        elif mode == "gray":
            return self.rgbs2grays()
        elif mode == "gray_normalized": # 0 - 1
            return self.rgbs2grays() / 255
        elif mode == "both_rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb])
        elif mode == "both_rgb_normal":
            return (self.current_inertial_state, self.rgbs2srgb_normal())
        elif mode == "both_rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba])
        elif mode == "both_gray":
            return (self.current_inertial_state, self.rgbs2grays())
        elif mode == "both_gray_normal":
            return (self.current_inertial_state, self.rgbs2grays() / 255)
        else:
            print("invalid Mode!")
    
    def rgbs2rgbs_normal(self):
        rgbs_norm = []
        num_imgs = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int))
        for i in range(num_imgs):
            rgbs_norm.append(self.rgb2rgb_normal(self.images_rgb[:,:,self.image_mask_rgb[3*i:3*(i+1)]]))
        rgbs_normal_cube = rgbs_norm[0]
        for i in range(num_imgs - 1):
            rgbs_normal_cube = np.dstack((rgbs_normal_cube, rgbs_norm[i+1]))
        return rgbs_normal_cube
        
        return np.array(self.images_rgb[:,:, self.image_mask_rgb] - np.atleast_3d(np.mean(self.images_rgb[:,:,self.image_mask_rgb], axis = 2)), dtype = np.float32) / np.atleast_3d(np.std(self.images_rgb[:,:,self.image_mask_rgb], axis = 2) + .001)
    
    def rgb2rgb_normal(self, rgb):
        # Special function to turn the RGB cube into a gray scale cube:
        return np.array(rgb - np.atleast_3d(np.mean(rgb, axis = 2)), dtype = np.float32) / np.atleast_3d(np.std(rgb, axis = 2) + .001)
    
    # Returns all grays from rgb, given your environment settings
    # Works on the internal 2 dim stacked vision cube images_rgb/a
    def rgbs2grays(self):
        grays = []
        num_imgs = np.sum(np.array(self.image_mask_FC_FR_FL, dtype = np.int))
        for i in range(num_imgs):
            grays.append(self.rgb2gray(self.images_rgb[:,:,self.image_mask_rgb[3*i:3*(i+1)]]))
        graycube = grays[0]
        for i in range(num_imgs - 1):
            graycube = np.dstack((graycube, grays[i+1]))
        return graycube
    
    def rgb2gray(self, rgb, isGray3D = True):
        # Special function to turn the RGB cube into a gray scale cube:
        if isGray3D:
            return np.array(np.atleast_3d(np.mean(rgb, axis = 2)), dtype = np.uint8)
        else:
            return np.array(np.mean(rgb, axis = 2), dtype = np.uint8)
    
    def get_new_images(self):
        # Wait to grad images
        tic = time.time()
        if (self.image_mask_FC_FR_FL[0] and  self.image_mask_FC_FR_FL[1] and  self.image_mask_FC_FR_FL[2]): 

            images = self.client.simGetImages([
            client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
            client.ImageRequest("1", client.ImageType.Scene, False, False), # Front Right
            client.ImageRequest("2", client.ImageType.Scene, False, False)]) # Front Left
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
            
            self.images_rgba = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
            self.images_rgb = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
            self.time_to_grab_images = time.time() - tic
            print("Time to Grab Images: ", self.time_to_grab_images)
             
        # We Just want front        
        elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)])
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            self.images_rgba = img_rgba_FC
            self.images_rgb = img_rgb_FC
            self.time_to_grab_images = time.time() - tic
            print("Time to Grab Images: ", self.time_to_grab_images)
        
        else:
            print("A screw up in set new images")

    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        # Get Base Inertial States
        tic = time.time()
        state = self.client.simGetGroundTruthKinematics()
        pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
        vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'], -1*state['linear_velocity']['z_val'])
        acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
        orien = (state['orientation']['x_val'],state['orientation']['y_val'],state['orientation']['z_val'], state['orientation']['w_val'])
        angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
        angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])

        
        # Store the current state
        self.current_inertial_state = np.array([pos[0], 
                                                pos[1], 
                                                pos[2]*-1,
                                                vel[0],
                                                vel[1],
                                                vel[2],
                                                acc[0], acc[1], acc[2],
                                                orien[0], orien[1], orien[2],
                                                angVel[0], angVel[1], angVel[2],
                                                angAcc[0], angAcc[1], angAcc[2]])
        self.time_to_grab_states = time.time() - tic
        print('Time to grab states: ', self.time_to_grab_states)
        # Load new images into env from sim
        self.get_new_images()
        # Store X and Y position information as well

    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        tic = time.time()
        # Either move by velocity, or move by angle at each timestep
        mbv, mba, mbh, _ = self.get_RC_action(act_num = action)
        quadState = self.client.simGetGroundTruthKinematics()
        if mbv is not None: #  Move By Velocity
            quad_vel = quadState['linear_velocity']
            self.client.moveByVelocityAsync(quad_vel['x_val'] + mbv[0], quad_vel['y_val'] + mbv[1], quad_vel['z_val'] + mbv[2], self.action_duration)
            time.sleep(self.action_duration)
            
        elif mba is not None: # Move By Angle
            quad_pry = quadState['orientation']
            self.client.moveByAngleZAsync(quad_pry['y_val'] + mba[0], quad_pry['x_val'] + mba[1], quad_pry['z_val'], quad_pry['w_val'] + mba[2], self.action_duration)
            time.sleep(self.action_duration)
        elif mbh is not None: # Move By Hover
            print("HOVER MODE EXECUTED")
            self.client.hoverAsync()
            time.sleep(1)
        else:
            print("error in do action")
        self.time_to_do_action = time.time() - tic
    # The reward function is broken into an intervention signal and non intervention signal, as the actions have different
        # state dependencies
    def calc_reward(self):
        tic = time.time()
        collision_info = self.client.simGetCollisionInfo()
        reward, done = self.reward_function(collision_info, self.current_inertial_state,self.max_altitude, self.min_altitude)
        self.time_to_calc_reward = time.time() - tic
        return (reward, done)
        
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

    def action_name(self, actionNum):
        dic = {0: 'No Action', 1: 'Hover', 2: 'Vx',3: 'Vy',
               4: 'Vz', 5: '-Vx', 6: '-Vy', 7: '-Vz',
               8: '+Roll', 9: '+Pitch', 10: '+Yaw', 11: '-Roll',
               12: '-Pitch', 13: '-Yaw'}
        return str(dic[actionNum])
    def action_num(self, actionName):
        dic = {'No Action': 0, 'Hover': 1, 'Vx': 2,'Vy' : 3,
               'Vz': 4, '-Vx': 5, '-Vy': 6 , '-Vz' : 7,
               '+Roll' : 8,'+Pitch' : 9, '+Yaw': 10, '-Roll': 11,
               '-Pitch' : 12, '-Yaw': 13}
        return str(dic[actionName])
    
    def reset(self):
        # Reset the Copter
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        #self.client.simSetVehiclePose(Vector3r(0,0,-2), ignore_collison = True)

        # Quickly raise the quadcopter to a few meters -- Speed up
        for i in range(8):
            state = self.client.simGetGroundTruthKinematics()
            quad_vel = state['linear_velocity']
            quad_offset = (0, 0, -self.scaling_factor)
            self.client.moveByVelocityAsync(quad_vel['x_val']+quad_offset[0], quad_vel['y_val']+quad_offset[1], quad_vel['z_val'] + quad_offset[2], self.action_duration)
            time.sleep(self.action_duration)
        
        state = self.client.simGetGroundTruthKinematics()
        self.initial_position = (state['position']['x_val'],
               state['position']['y_val'],
               state['position']['z_val']*-1)
        
        self.initial_velocity = (state['linear_velocity']['x_val'],
               state['linear_velocity']['y_val'],
               state['linear_velocity']['z_val'])
        
        print("Initial Quad Position: ", self.initial_position)
        print('Reset Complete')
        
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        self.extra_metadata = {'action': 0, 'action_name': 0, 'env_state': {'resetting': True, 'running': False},
                               'mode': 0, 'reward': 0, 'done': 0, 'times': {'act_time': 0,
                                                            'sim_img_time': 0,
                                                            'sim_state_time': 0,
                                                            'reward_time': 0,
                                                            'step_time': 0}}
        
        if self.mode == "inertial":
            return (self.current_inertial_state, self.extra_metadata)
        elif self.mode == "rgb":
            # Reduce dimentionality of obs
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgb[:,:,self.image_mask_rgb], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, self.extra_metadata)
        elif self.mode == "rgb_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, self.extra_metadata)
        elif self.mode == "rgba":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, self.extra_metadata)
        elif self.mode == "gray":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, self.extra_metadata)
        elif self.mode == "gray_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays() / 255, pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.obs4, self.extra_metadata)
        elif self.mode == "both_rgb":
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgb[:,:, self.image_mask_rgb], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2rgbs_normal(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, self.extra_metadata)
        elif self.mode == "both_rgba": 
            self.obs4 = trim_append_state_vector(self.obs4, self.images_rgba[:,:, self.image_mask_rgba], pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, self.extra_metadata)
        elif self.mode == "both_gray":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays(), pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            self.obs4 = trim_append_state_vector(self.obs4, self.rgbs2grays()/ 255, pop_index = self.IMG_VIEWS * self.IMG_CHANNELS) # pop old and append new state obsv
            return (self.current_inertial_state, self.obs4, self.extra_metadata)
        else:
            print("invalid Mode!")
        



