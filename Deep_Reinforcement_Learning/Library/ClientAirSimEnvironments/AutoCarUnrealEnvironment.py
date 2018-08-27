# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:51:21 2018

@author: natsn
"""

import numpy as np
import time
import sys
import multiprocessing 
import threading
from RewardingFunctions import car_racing_rewarding_function
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Library")
from airsim import client
from airsim.types import Vector3r
import AirSimGUI
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util")
from ImageProcessing import trim_append_state_vector

        
# The AirSim Environment Class for Quadrotor Safety
class AutoCarUnrealEnvironment:
    # image mask is what images you'd like to return from the simulator
    # reward function sets how the car is incentivized to move
    # mode defines what the environment will return ( ie: states, images, or both)
    
    def __init__(self, vehicle_name = "Car1", 
                 action_duration = .08,
                 image_mask_FC_FR_FL = [True, True, True],
                 sim_mode = "rgb",
                 IMG_HEIGHT = 128,
                 IMG_WIDTH = 128,
                 IMG_STEP = 3,
                 reward_function = car_racing_rewarding_function):
        
        # Set reward function
        self.reward_function = reward_function
        
        # Set Environment Return options
        self.action_duration = action_duration
        self.mode = sim_mode
        
        # Set Drive Commands to zero initially
        self._throttle = 0 # Used as a constant gain factor for the action throttle. 
        self._steering = 0 # Each action lasts this many seconds
        self._brake  = 0
        
        self.THROTTLE_INC = .10
        self.THROTTLE_DEC = -.10
        self.BRAKE_INC = .10
        self.BRAKE_DEC = -.20
        self.STEER_LEFT_INC = -.10
        self.STEER_RIGHT_INC = .10
        
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
        
        
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 15 # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz 
        
        # Throttle up, throttle down, increase brake, decrease break, left_steer, right_steer, No action
        self.count_car_actions = 7 
        
        # Initialize the current inertial state to zero
        self.current_inertial_state = np.array(np.zeros(self.count_inertial_state_variables))

        # Initialize the IMAGE variables -- We Take in Front Center, Right, Left
        self.images_rgb = None
        self.images_rgba = None
        self.image_mask_rgb = np.array([ [0+3*i,1+3*i,2+3*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_rgba = np.array([ [0+4*i,1+4*i,2+4*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_FC_FR_FL = image_mask_FC_FR_FL
        
        # Connect to the AirSim simulator and begin:
        print('Initializing Car Client')
        self.client = client.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        print('Initialization DONE!')
        
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
        self.app = AirSimGUI.CarGUI(GUIConn, vehicle_names = [vehicle_name],
                                                   num_video_feeds = num_video_feeds, isRGB = isRGB, isNormal = isNormal)
        
        # Timing Operations Initialize
        self.time_to_do_action = 0
        self.time_to_grab_images = 0
        self.time_to_grab_states = 0
        self.time_to_calc_reward = 0
        self.time_to_step = 0
        self.extra_metadata = None

    # This is the List of all possible actions for the Car:
    # throttle_up = +.1 (speed up)
    # throttle_down = -.1 (speed down)
    # steer_left = -.05
    # steer_right = +.05
    # brake_up = +.1
    # brake_down = -.1
    # no action (NOP)
    
    # the action is an integer value corresponding to the above action choices
    # Three step modes: 
    # 1: Inertial: Returns the quadcopter's inertial state information from the simulator
    # 2: Image: Returns the quadcopter's image information from the simulator
    
    def step(self, action):
        tic = time.time()
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        reward, done = self.calc_reward()
        
        self.time_to_step = time.time() - tic
        print('Time to step: ', self.time_to_step)
        
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
        
        self.extra_metadata = {'action': action, 'action_name': self.action_name(action), 'env_state': {'resetting': False, 'running': True},
                               'mode': self.mode, 'reward': reward, 'done': done, 'times': {'act_time': self.time_to_do_action,
                                                            'sim_img_time': self.time_to_grab_images,
                                                            'sim_state_time': self.time_to_grab_states,
                                                            'reward_time': self.time_to_calc_reward,
                                                            'step_time': self.time_to_step}}
        
        self.gui_data['state'] = self.current_inertial_state
        self.gui_data['obs'] = self.obs4
        self.gui_data['meta'] = self.extra_metadata
        t_gui = threading.Thread(target = self.simEnvDataConn.send, args = ({self.vehicle_name : self.gui_data},))
        t_gui.start()
        t_gui.join()
    
    
    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        tic = time.time()
        # Get Base Inertial States
        state = self.client.simGetGroundTruthKinematics()
        pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
        vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'],state['linear_velocity']['z_val'])
        acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
        angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
        angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
        
        # Store the current state
        self.current_inertial_state = np.array([pos[0] - self.initial_position[0], 
                                                pos[1] - self.initial_position[1],
                                                pos[2] - self.initial_position[2],
                                                vel[0] - self.initial_velocity[0],
                                                vel[1] - self.initial_velocity[1],
                                                vel[2] - self.initial_velocity[2],
                                                acc[0], acc[1], acc[2],
                                                angVel[0], angVel[1], angVel[2],
                                                angAcc[0], angAcc[1], angAcc[2]])
        self.time_to_grab_states = time.time() - tic
        print('Time to grab states: ', self.time_to_grab_states)
        
        self.get_new_images()    
        print("Time to Grab All Images: ", self.time_to_grab_images)
        
    def get_new_images(self):
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
        tic = time.time()
        self.client.simPause(True)
        if (self.image_mask_FC_FR_FL[0] and  self.image_mask_FC_FR_FL[1] and  self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
            client.ImageRequest("1", client.ImageType.Scene, False, False), # Front Right
            client.ImageRequest("2", client.ImageType.Scene, False, False)]) # Front Left
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            
            img1d_FR = np.fromstring(images[1].image_data_uint8, dtype=np.uint8) 
            img_rgba_FR = np.array(img1d_FR.reshape(images[1].height, images[1].width, 4), dtype = np.uint8)
            img_rgb_FR = img_rgba_FR[:,:,0:3]
            
            #plt.imshow(img_rgb_FR)
            #plt.show()
            #time.sleep(2)
            
            img1d_FL = np.fromstring(images[2].image_data_uint8, dtype=np.uint8) 
            img_rgba_FL = np.array(img1d_FL.reshape(images[2].height, images[2].width, 4), dtype = np.uint8)
            img_rgb_FL = img_rgba_FL[:,:,0:3]
            #plt.imshow(img_rgb_FL)
            #plt.show()
            #time.sleep(2)
            
            # Can either use the RGBA images or the RGB Images
            self.images_rgba = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
            self.images_rgb = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
            self.time_to_grab_images = time.time() - tic
            
        # We Just want front view      
        elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)]) # Front Center
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            self.images_rgba = img_rgba_FC
            self.images_rgb = img_rgb_FC
            self.time_to_grab_images = time.time() - tic
        self.client.simPause(False)
        
    
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
    
    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        tic = time.time()
        # Returns the increment or decrement (AKA the DQN's Action choice)
        throttle_cmd, breaking_cmd, steering_cmd, _ = self.get_RC_action(act_num = action)
        
        #print("thr " + throttle_cmd + "Bra " + breaking_cmd + "str " + steering_cmd )
        self.brake = self.brake + breaking_cmd
        self.steering = self.steering + steering_cmd
        self.throttle = self.throttle + throttle_cmd
        #print("Desired Steering Commands (TBS): ", self.throttle, self.brake, self.steering)
        
        car_controls = client.CarControls()
        car_controls.throttle = self.throttle
        car_controls.steering = self.steering
        car_controls.brake = self.brake
        self.client.setCarControls(car_controls) # Send off to the simulator!
        time.sleep(self.action_duration)
        self.time_to_do_action = time.time() - tic
        print('Time to do action: ', self.time_to_do_action)
    
    # The Reward function is broken into an intervention signal and non intervention signal, as the actions have different
    # State Dependencies
    def calc_reward(self):
        collision_info = self.client.simGetCollisionInfo()
        reward, done = self.reward_function(collision_info, self.current_inertial_state)
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
        # Else, use a random action value and return it (used for simulating semi - autonomous modes)
        else:
            rand_act_n = np.random.randint(self.count_car_actions)
            
        throttle_cmd = 0
        brake_cmd = 0
        steering_cmd = 0

        if rand_act_n == 0: # Throttle Up
            throttle_cmd = self.THROTTLE_INC
        elif rand_act_n == 1: # Throttle Down
            throttle_cmd = self.THROTTLE_DEC
        elif rand_act_n == 2: # Increase Break
            brake_cmd = self.BRAKE_INC
        elif rand_act_n == 3: # Decrease Break
            brake_cmd = self.BRAKE_DEC
        elif rand_act_n == 4: # Steer Left
            steering_cmd = self.STEER_LEFT_INC
        elif rand_act_n == 5: # Steer Right
            steering_cmd = self.STEER_RIGHT_INC
        else:
            # No action
            pass
        # Move by angle or move by velocity (one of the two will be set to None), meaning we either move by one or the other to generate a trajectory
        return throttle_cmd, brake_cmd, steering_cmd, rand_act_n
        
    @property
    def throttle(self):
        return self._throttle
    @property
    def brake(self):
        return self._brake
    @property
    def steering(self):    
        return self._steering
    @steering.setter
    def steering(self, s):
        if (s <= 1 and s >= -1):
            self._steering = s
            #print("Steering: ", self._steering)
        elif s < -1:
            #print("Steering Value too low")
            self._steering = -1
        else: #s>1
            #print("Steering Value too high")
            self._steering = 1 
    @throttle.setter
    def throttle(self, t):
        if (t <= 1 and t >= 0):
            if t > self.throttle:
                self._brake = 0
            self._throttle = t
            
            #print("Throttle: ", self._throttle)
        elif t < 0:
            #print("Throttle Value too low")
            self._throttle = 0
        else: # >1 
            #print("Throttle Value too high")
            self._throttle = 1
            self._brake = 0
    @brake.setter
    def brake(self, b):
        if (b <= .5 and b >= 0):
            self._brake = b
            #print("Break: ", self._brake)
            #self._throttle = 0
        elif b < 0:
            #print("Break Value too low")
            self._brake = 0
        else: #b>1
            #print("Break Value too high")
            self._brake = .5
            #self._throttle = 0
            
    def action_name(self, actionName):
        dic = {0: 'Throttle Up', 1: 'Throttle Down', 2: 'Brake Up',
               3: 'Brake Down', 4: 'Steer Left', 5: 'Steer Right', 6: 'No Action'}
        return dic[actionName]
    def action_num(self, actionNum):
        dic = {'Throttle Up' : 0, 'Throttle Down' : 1, 'Brake Up' : 2,
               'Brake Down' : 3, 'Steer Left' : 4, 'Steer Right': 5, 'No Action': 6}
        return dic[actionNum]
    
    def reset(self):
        # Reset the Car
        print('Reseting Car')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        time.sleep(.5) # to make sure acceleration doesnt come back insane the first loop
        print('Reset Complete')
        # Start Timer for episode's first step:
        #state = self.client.getCarState()
        state = self.client.simGetGroundTruthKinematics()
        self.initial_position = (state['position']['x_val'],
               state['position']['y_val'],
               state['position']['z_val'])
        
        self.initial_velocity = (state['linear_velocity']['x_val'],
               state['linear_velocity']['y_val'],
               state['linear_velocity']['z_val'])
        
        self.dt = 0
        self.tic = time.time()
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        # Reset throttle break and steer
        self.throttle = 0
        self.steering = 0
        self.brake = 0
        
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






