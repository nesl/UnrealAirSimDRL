# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:52:05 2018

@author: natsn
"""

import numpy as np
import time
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")
from airsim import client
from airsim.types import Vector3r, Quaternionr
import sys
from ImageProcessing import trim_append_state_vector
from RewardingFunctions import car_racing_rewarding_function



# The AirSim Environment Class for Multi Cars Concurrently in simulator
class MultiCarUnrealEnvironment:
    # image mask is what images you'd like to return from the simulator
    # reward function sets how the car is incentivized to move
    # mode defines what the environment will return ( ie: states, images, or both)
    
    def __init__(self, vehicle_names, image_mask_FC_FR_FL = [True, False, False],
                 reward_function = car_racing_rewarding_function, mode = "rgb_normal"):
        self.vehicle_names = vehicle_names
        # Set reward function
        self.reward_function = reward_function
        
        # Set Environment Return options
        self.mode = mode
        self.mode
        
        # Set Drive Commands to zero initially
        self._throttle = dict.fromkeys(self.vehicle_names, 0) # Used as a constant gain factor for the action throttle. 
        self._steering = dict.fromkeys(self.vehicle_names, 0) # Each action lasts this many seconds
        self._brake  = dict.fromkeys(self.vehicle_names, 0)
        
        self.THROTTLE_INC = .10
        self.THROTTLE_DEC = -.10
        self.BRAKE_INC = .10
        self.BRAKE_DEC = -.20
        self.STEER_LEFT_INC = -.10
        self.STEER_RIGHT_INC = .10
        
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 15 # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz 
        
        # Throttle up, throttle down, increase brake, decrease break, left_steer, right_steer, No action
        self.count_car_actions = 7 
        
        # Initialize the current inertial state to zero
        self.current_inertial_states = dict.fromkeys(self.vehicle_names, np.array(np.zeros(self.count_inertial_state_variables)))

        # Initialize the IMAGE variables -- We Take in Front Center, Right, Left
        self.images_rgb = dict.fromkeys(self.vehicle_names, 0)
        self.images_rgba = dict.fromkeys(self.vehicle_names, 0)
        self.image_mask_rgb = np.array([ [0+3*i,1+3*i,2+3*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_rgba = np.array([ [0+4*i,1+4*i,2+4*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_FC_FR_FL = image_mask_FC_FR_FL
        
        self.initial_position = dict.fromkeys(self.vehicle_names, 0)
        self.initial_velocity = dict.fromkeys(self.vehicle_names, 0)
        
        # Connect to the AirSim simulator and begin:
        print('Initializing Car Client')
        self.client = client.CarClient()
        self.client.confirmConnection()
        for vn in self.vehicle_names:
            self.client.enableApiControl(True, vn)
            orien = Vector3r(0, 0, 0)
            self.client.simSetCameraOrientation(0, orien.to_Quaternionr(), vehicle_name = vn) #radians
            orien = Vector3r(0, .12, -np.pi/9)
            self.client.simSetCameraOrientation(1, orien, vehicle_name = vn)
            orien = Vector3r(0, .12, np.pi/9)
            self.client.simSetCameraOrientation(2, orien, vehicle_name = vn)
        # Reset Collion Flags
        print('Initialization Complete!')
        # Timing Operations Initialize
        self.dt = 0
        self.tic = 0
        self.toc = 0

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
    
    def step(self, actions):
        
        # 1. Take action in the simulator based on the agents action choice
        self.do_actions(actions)
        
        # 2: Update the environment state variables after the action takes place
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
            return (self.rgbs2grays() / 255, rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_states, self.get_rgbs(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2rgbs_normal(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays(), rewards, dones, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays()/ 255, rewards, dones, self.extra_metadata)
        else:
            print("invalid Mode!")
            
    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        self.client.simPause(True)
        # Get Base Inertial States
        
        # Collect and Display Elapsed Time Between States
        self.toc = time.time()
        self.dt = self.toc - self.tic
        for vn in self.vehicle_names:
            state = self.client.simGetGroundTruthKinematics(vehicle_name = vn)
            pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
            vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'],state['linear_velocity']['z_val'])
            acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
            angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
            angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
            
            # Store the current state
            self.current_inertial_states[vn] = np.array([pos[0] - self.initial_position[vn][0], 
                                                    pos[1] - self.initial_position[vn][1],
                                                    pos[2] - self.initial_position[vn][2],
                                                    vel[0] - self.initial_velocity[vn][0],
                                                    vel[1] - self.initial_velocity[vn][1],
                                                    vel[2] - self.initial_velocity[vn][2],
                                                    acc[0], acc[1], acc[2],
                                                    angVel[0], angVel[1], angVel[2],
                                                    angAcc[0], angAcc[1], angAcc[2]])
        
        # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz 
        
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
        for vn in self.vehicle_names:
            if (self.image_mask_FC_FR_FL[0] and  self.image_mask_FC_FR_FL[1] and  self.image_mask_FC_FR_FL[2]): 
                images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
                client.ImageRequest("1", client.ImageType.Scene, False, False), # Front Right
                client.ImageRequest("2", client.ImageType.Scene, False, False)], vehicle_name = vn) # Front Left
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
                self.images_rgba[vn] = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
                self.images_rgb[vn] = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
                print("Time to Grab All Images: ", time.time() - self.toc)
                
            # We Just want front view      
            elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
                images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)], vehicle_name = vn) # Front Center
                img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
                img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
                img_rgb_FC = img_rgba_FC[:,:,0:3]
                
                self.images_rgba[vn] = img_rgba_FC
                self.images_rgb[vn] = img_rgb_FC
                print("Time to Grab Images: ", time.time() - self.toc)
            
        
        self.client.simPause(False)
        # Store X and Y position information as well
        self.extra_metadata = self.dt
        # Start the timer again as the next state is saved. Return the current state and meta-data 
        self.tic = time.time()
    
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
            return self.rgbs2grays() / 255
        elif mode == "both_rgb":
            return (self.current_inertial_states, self.get_rgbs())
        elif mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2srgb_normal())
        elif mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas())
        elif mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays())
        elif mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays() / 255)
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
        
    #Ask the simulator to perform control command or to block control command
    def do_actions(self, actions):
        
        # Returns the increment or decrement (AKA the DQN's Action choice)
        for vn, act in zip(self.vehicle_names, actions):
            throttle_cmd, breaking_cmd, steering_cmd, _ = self.get_RC_action(act_num = act)
            
            #print("thr " + throttle_cmd + "Bra " + breaking_cmd + "str " + steering_cmd )
            self.set_brake(vn, self.get_brake(vn) + breaking_cmd)
            self.set_steering(vn, self.get_steering(vn) + steering_cmd)
            self.set_throttle(vn, self.get_throttle(vn) + throttle_cmd)
            #print("Desired Steering Commands (TBS): ", self.throttle, self.brake, self.steering)
            
            car_controls = client.CarControls()
            car_controls.throttle = self.get_throttle(vn)
            car_controls.steering = self.get_steering(vn)
            car_controls.brake = self.get_brake(vn)
            self.client.setCarControls(car_controls, vehicle_name = vn) # Send off to the simulator!
        time.sleep(.08)
    # The reward function is broken into an intervention signal and non intervention signal, as the actions have different
        # state dependencies
    def calc_rewards(self):
        rewards = []
        dones = []
        for vn in self.vehicle_names:
            collision_info = self.client.simGetCollisionInfo(vehicle_name = vn)
            r, d = self.reward_function(collision_info, self.current_inertial_states[vn])
            rewards.append(r)
            dones.append(d)
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
        
    def get_throttle(self, vehicle_name):
        return self._throttle[vehicle_name]
    
    def get_brake(self, vehicle_name):
        return self._brake[vehicle_name]

    def get_steering(self, vehicle_name):    
        return self._steering[vehicle_name]

    def set_steering(self, vehicle_name, val):
        if (val <= 1 and val >= -1):
            self._steering[vehicle_name] = val
            #print("Steering: ", self._steering)
        elif val < -1:
            #print("Steering Value too low")
            self._steering[vehicle_name] = -1
        else: #s>1
            #print("Steering Value too high")
            self._steering[vehicle_name] = 1 
            
    def set_throttle(self, vehicle_name, t):
        if (t <= .8 and t >= 0):
            if t > self.throttle[vehicle_name]:
                self._brake[vehicle_name] = 0
            self._throttle[vehicle_name] = t
            
            #print("Throttle: ", self._throttle)
        elif t < 0:
            #print("Throttle Value too low")
            self._throttle[vehicle_name] = 0
        else: # >1 
            #print("Throttle Value too high")
            self._throttle[vehicle_name] = .8
            self._brake[vehicle_name] = 0
            
    def set_brake(self, vehicle_name, b):
        if (b <= .5 and b >= 0):
            self._brake[vehicle_name] = b
            #print("Break: ", self._brake)
            #self._throttle = 0
        elif b < 0:
            #print("Break Value too low")
            self._brake[vehicle_name] = 0
        else: #b>1
            #print("Break Value too high")
            self._brake[vehicle_name] = .5
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
        print('Reset Complete')
        self.client.confirmConnection()
        for vn in self.vehicle_names:
            self.client.enableApiControl(True, vn)
            state = self.client.simGetGroundTruthKinematics(vehicle_name = vn)
            self.initial_position[vn] = (state['position']['x_val'],
                   state['position']['y_val'],
                   state['position']['z_val'])
            
            self.initial_velocity[vn] = (state['linear_velocity']['x_val'],
                   state['linear_velocity']['y_val'],
                   state['linear_velocity']['z_val'])
        
        self.dt = 0
        self.tic = time.time()
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        # Reset throttle break and steer
        self._throttle = dict.fromkeys(self.vehicle_names, 0)
        self._steering = dict.fromkeys(self.vehicle_names, 0)
        self._brake = dict.fromkeys(self.vehicle_names, 0)
        
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
            return (self.rgbs2grays() / 255, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_states, self.get_rgbs(), self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_states, self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_states, self.get_rgbas(), self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_states, self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_states, self.rgbs2grays()/ 255, self.extra_metadata)
        else:
            print("invalid Mode!")


