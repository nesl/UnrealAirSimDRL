# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:26:01 2018

@author: natsn
"""

import numpy as np
import time
from airsim import client
from airsim.types import Vector3r, Quaternionr

# The drone is incentivized to use its vision to cruise around the world at 5 meters
# The faster the drone moves around the world, the more points it achieves

def drone_forest_racer_rewarding_function(collision_info, current_inertial_state, 
                                          max_altitude = 12, min_altitude = 2):
    mean_height = (max_altitude + min_altitude) / 2
    collided = False
    # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd, Rdd, Pdd, Ydd
    if  collision_info.has_collided:  # meters / second
        collided = True
        print("Collision Info: ", collision_info.has_collided)
        
    
    # 2. Check for limits:
    # If we are above our z threshold, throw done flag
    if current_inertial_state[2] > max_altitude or current_inertial_state[2] < min_altitude:
        done = True
        reward = -50
        print('Out of Bounds, Resetting Episode!')
        return reward, done

    # Check for collision with ground
    elif collided:
        reward = -200
        done = True
        print('COLLIDED: ', collided)
        return reward, done
    
    # If we havent collided and are within the min-max altitude bounds, 
    else:
        # Max is 1 if we are at mean height
        reward_height = -(current_inertial_state[2] - mean_height)**2 + 1
        reward_speed = np.tanh(.1*(np.sqrt(current_inertial_state[3]**2 + current_inertial_state[4]**2 + current_inertial_state[5]**2) - 5)) + .5*np.log10(current_inertial_state[3]**2 + .001) + .5*np.log10(current_inertial_state[4]**2 + .001)
        reward = reward_height + reward_speed 
        done = False
        return reward, done
    
    
# A function to keep the drone safe in semi-autonomous mode
def drone_safety_rewarding_function(collision_info, current_inertial_state, 
                                    action_choice, max_altitude = 15, min_altitude = .25):
    # 1) Determine if a collision has occured
    # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd
    
    collided = False
    if  collision_info.time_stamp or current_inertial_state[0] < min_altitude: # meters / second
        collided = True

    # 2. Check for limits:
    # If we are above our z threshold, throw done flag
    if current_inertial_state[2] > max_altitude:
        done = True
        reward = 0
        print('Out of Bounds, Resetting Episode!')
        return reward, done

    # Check for collision with ground
    if collided:
        reward = -100
        done = True
        print('COLLIDED: ', collided)
        return reward, done

    # Check for intervention
    if action_choice == 1: # Which is to intervene and hover
        reward = -5
        done = False
        return reward, done

    else: # The drone is kept safe and we can reward the algorithm
        reward = 2
        done = False
        return reward, done

# Car's reinforcement learning task here is to race as fast as possible without crashing
def car_racing_rewarding_function(collision_info, current_inertial_state, bias = 30):
# 1) Determine if a collision has occured

    collided = False
    reward = 0
    if  collision_info.time_stamp or collision_info.has_collided: # meters / second
        collided = True

    # Check for collision with something
    if collided:
        reward = -200
        done = True
        print('COLLIDED! ', collided)
        return reward, done

    else:
        # Thevelocity 
        reward = 6*np.tanh(0.1*(np.sqrt(current_inertial_state[3]**2 + current_inertial_state[4]**2) - 10)) + .25*np.log(.0001 + current_inertial_state[0]**2 +current_inertial_state[1]**2) + np.log(np.abs(current_inertial_state[3]) + .01) + np.log(np.abs(current_inertial_state[4]) + .01)
        done = False
        print("REWARD: ", reward)
        return reward, done

# The AirSim Environment Class for Quadrotor Safety
class QuadcopterUnrealEnvironment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    def __init__(self, max_altitude = 14,
                 min_altitude = 2,
                 image_mask_FC_FR_FL = [True, False, False], # Front Center, Front right, front left
                 reward_function = drone_forest_racer_rewarding_function,
                 mode = "both_rgb"):
        
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
    def step(self, action):
        
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        reward, done = self.calc_reward()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, reward, done, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(),reward, done, self.extra_metadata)
        elif self.mode == "rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), reward, done, self.extra_metadata)
        elif self.mode == "gray_normal":
            return (self.rgbs2grays() / 255, reward, done, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_state, self.rgbs2rgbs_normal(), reward, done, self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_state, self.rgbs2grays(), reward, done, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_state, self.rgbs2grays()/ 255, reward, done, self.extra_metadata)
        else:
            print("invalid Mode!")
    
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
        print("Collecting Images from AirSim")
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
            print('Images Collected!')
            print("Time to Grab Images: ", time.time() - tic)
             
        # We Just want front        
        elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)])
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            self.images_rgba = img_rgba_FC
            self.images_rgb = img_rgb_FC
            print('Images Collected!')
            print("Time to Grab Images: ", time.time() - tic)
        
        else:
            print("A screw up in set new images")

    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        
        # Get Base Inertial States
        state = self.client.simGetGroundTruthKinematics()
        pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
        vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'], -1*state['linear_velocity']['z_val'])
        acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
        orien = (state['orientation']['x_val'],state['orientation']['y_val'],state['orientation']['z_val'], state['orientation']['w_val'])
        angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
        angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
        
        # Collect and Display Elapsed Time Between States
        self.toc = time.time()
        self.dt = self.toc - self.tic
        
        # Store the current state
        self.current_inertial_state = np.array([pos[0], 
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
        self.get_new_images()
        # Store X and Y position information as well
        self.extra_metadata = self.dt
        # Start the timer again as the next state is saved. Return the current state and meta-data 
        self.tic = time.time()
        
    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        
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
            self.client.hoverAsync()
            time.sleep(.75)
        else:
            print("error in do action")
    # The reward function is broken into an intervention signal and non intervention signal, as the actions have different
        # state dependencies
    def calc_reward(self):
        collision_info = self.client.simGetCollisionInfo()
        reward, done = self.reward_function(collision_info, self.current_inertial_state,self.max_altitude, self.min_altitude)
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
        # Reset the Copter
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Quickly raise the quadcopter to a few meters -- Speed up
        for i in range(15):
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
        
        # Start Timer for episode's first step:
        self.dt = 0
        self.tic = time.time()
        time.sleep(.5)
        
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "gray_normal":
            return (self.rgbs2grays() / 255, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_state, self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_state, self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_state, self.rgbs2grays()/ 255, self.extra_metadata)
        else:
            print("invalid Mode!")
        



class RandomDroneOperator:
    
    def __init__(self, env):
        self.num_drone_movements = 14
        self.current_action = 0
        
        # Set the current
        self.current_action = self.get_random_action()

    # Leaving this blank returns you a random action with its corresponding mbv/mba/mbh
    # Filling in which action you desire here returns its corresponding mba/mbv/mbh 
    def get_probabilistic_action(self, prob_of_new_action_selected = .25):
        randnum = np.random.rand()
        if randnum > (1 - prob_of_new_action_selected):
            self.current_action = self.get_random_action()
        return self.current_action
        
    def get_random_action(self):
        # Random drone action to be selected
        rand_act_n = np.random.randint(0,self.num_drone_movements)
        # The operators action choice
        return rand_act_n
        
    def read_in_action_sequence():
        pass # We'd like to take a file of control actions and read them into the drone for control
        


# The AirSim Environment Class for Quadrotor Safety
class CarUnrealEnvironment:
    # image mask is what images you'd like to return from the simulator
    # reward function sets how the car is incentivized to move
    # mode defines what the environment will return ( ie: states, images, or both)
    
    def __init__(self, image_mask_FC_FR_FL = [True, False, False],
                 reward_function = car_racing_rewarding_function, mode = "rgb"):
        
        # Set reward function
        self.reward_function = reward_function
        
        # Set Environment Return options
        self.mode = mode
        self.mode
        
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
        print('Initialization Complete!')
        
        print("Setting Camera Views")
        orien = Vector3r(0, 0, 0)
        self.client.simSetCameraOrientation(0, orien.to_Quaternionr()) #radians
        orien = Vector3r(0, .12, -np.pi/9)
        self.client.simSetCameraOrientation(1, orien)
        orien = Vector3r(0, .12, np.pi/9)
        self.client.simSetCameraOrientation(2, orien)
        # Reset Collion Flags
        print("Setting Camera Views DONE")
        time.sleep(1)
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
    
    def step(self, action):
        
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        reward, done = self.calc_reward()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, reward, done, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(),reward, done, self.extra_metadata)
        elif self.mode == "rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), reward, done, self.extra_metadata)
        elif self.mode == "gray_normal":
            return (self.rgbs2grays() / 255, reward, done, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_state, self.rgbs2rgbs_normal(), reward, done, self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_state, self.rgbs2grays(), reward, done, self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_state, self.rgbs2grays()/ 255, reward, done, self.extra_metadata)
        else:
            print("invalid Mode!")
            
    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        
        # Get Base Inertial States
        state = self.client.simGetGroundTruthKinematics()
        pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
        vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'],state['linear_velocity']['z_val'])
        acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
        angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
        angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
        
        # Collect and Display Elapsed Time Between States
        self.toc = time.time()
        self.dt = self.toc - self.tic
        
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
        
        # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz 
        
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
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
            print("Time to Grab All Images: ", time.time() - self.toc)
            
        # We Just want front view      
        elif (self.image_mask_FC_FR_FL[0] and not self.image_mask_FC_FR_FL[1] and not self.image_mask_FC_FR_FL[2]): 
            images = self.client.simGetImages([client.ImageRequest("0", client.ImageType.Scene, False, False)]) # Front Center
            img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
            img_rgba_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
            img_rgb_FC = img_rgba_FC[:,:,0:3]
            
            self.images_rgba = img_rgba_FC
            self.images_rgb = img_rgb_FC
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
        time.sleep(.08)
    # The reward function is broken into an intervention signal and non intervention signal, as the actions have different
        # state dependencies
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
        if (t <= .8 and t >= 0):
            if t > self.throttle:
                self._brake = 0
            self._throttle = t
            
            #print("Throttle: ", self._throttle)
        elif t < 0:
            #print("Throttle Value too low")
            self._throttle = 0
        else: # >1 
            #print("Throttle Value too high")
            self._throttle = .8
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
        
        if self.mode == "inertial":
            return (self.current_inertial_state, self.extra_metadata)
        elif self.mode == "rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "rgb_normal":
            return (self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "gray":
            return (self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "gray_normal": # 0 - 1
            return (self.rgbs2grays() / 255, self.extra_metadata)
        elif self.mode == "both_rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "both_rgb_normal":
            return (self.current_inertial_state, self.rgbs2rgbs_normal(), self.extra_metadata)
        elif self.mode == "both_rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "both_gray":
            return (self.current_inertial_state, self.rgbs2grays(), self.extra_metadata)
        elif self.mode == "both_gray_normal":
            return (self.current_inertial_state, self.rgbs2grays() / 255.0, self.extra_metadata)
        else:
            print("invalid Mode!")









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




'''
# Use this class to get real physical data from the drone for your DQN
class QuadrotorSensoryDroneKitEnvironment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    def __init__(self, minimax_z_thresh):

        # States will be pitch, roll, vx,vy,vz
        # Number of states = 4 states, + the 'RC' simulated action

        self.num_user_RC_actions = 13 # 6 angular, 6 linear, 1 no-move
        self.num_state_variables = 7 # posz, velz, roll, pitch
        self.initial_position = 0
        self.Lgain = .2 #.2,.3
        self.Ugain = .4 #.4,.5
        self.low_count = 0
        self.target_alt = 4 # meters
        self.count = 0
        
        self.total_state_variables = self.num_user_RC_actions + self.num_state_variables
        s = np.zeros(self.total_state_variables)
        self.current_state = np.array(s)

        self.current_action = 5
        self.previous_action = 4

        self.scaling_factor = .35 # Hyper-parameter up for changing
        self.dt = 0
        self.duration = .20 # initially

        #assert minimax_z_thresh[0] >= 0 and minimax_z_thresh[1] >= 0 and minimax_z_thresh[0] < minimax_z_thresh[1]
        self.minimax_z_thresh = minimax_z_thresh

        self.lb = pp.LabelBinarizer() # Binarizer for making categorical RC action input
        self.lb.fit(range(self.num_user_RC_actions)) # actions 0-12

        # connect to the dronekit API
        print('Initializing Vehicle')
        self.vehicle = connect('udpin:0.0.0.0:14550', wait_ready = True)
        print('Initialization Complete!')

    # The action will be to Go Left / Right / Up / Down for 20 ms
    def step(self, action):
        # If the action is to intervene, we must use last states state info,
        # since after the intervention we will always be stable

        # 1. Take action in the simulator based on the DRL predict function -- 1 or 0
        intervened = self.do_action(action)
        print(intervened)

        # 2: Calculate next state
        self.current_state, extra_metadata = self.pget_simulator_state_info()

        # 3: Calculate reward
        # reward, done = self.calc_reward(self.current_state,intervened)
        
        if self.current_state[2] < 0:
            done = True
            self.set_drone_velocity(0,0,0,2)
        else:
            done = False

        return self.current_state, done, extra_metadata

    def set_drone_velocity(self,vx,vy,vz,duration):
        """
        Move vehicle in direction based on specified velocity vectors.
        """
        print("Velocities are Vx: ", vx, ", Vy: ", vy, "zz: ", vz)
        msg = self.vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, # lat_int - X Position in WGS84 frame in 1e7 * meters
        0, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0, # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        vx, # X velocity in NED frame in m/s
        vy, # Y velocity in NED frame in m/s
        vz, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

        self.vehicle.send_mavlink(msg)
        # Sleep for durations length
        time.sleep(duration)

    def to_quaternion(self, roll = 0.0, pitch = 0.0, yaw = 0.0, rad = False):
        """
        Convert degrees to quaternions
        """
        if not rad: # in degrees
            t0 = math.cos(math.radians(yaw * 0.5))
            t1 = math.sin(math.radians(yaw * 0.5))
            t2 = math.cos(math.radians(roll * 0.5))
            t3 = math.sin(math.radians(roll * 0.5))
            t4 = math.cos(math.radians(pitch * 0.5))
            t5 = math.sin(math.radians(pitch * 0.5))

            w = t0 * t2 * t4 + t1 * t3 * t5
            x = t0 * t3 * t4 - t1 * t2 * t5
            y = t0 * t2 * t5 + t1 * t3 * t4
            z = t1 * t2 * t4 - t0 * t3 * t5

            return [w, x, y, z]

        else: # already in radians
            t0 = math.cos((yaw * 0.5))
            t1 = math.sin((yaw * 0.5))
            t2 = math.cos((roll * 0.5))
            t3 = math.sin((roll * 0.5))
            t4 = math.cos((pitch * 0.5))
            t5 = math.sin((pitch * 0.5))

            w = t0 * t2 * t4 + t1 * t3 * t5
            x = t0 * t3 * t4 - t1 * t2 * t5
            y = t0 * t2 * t5 + t1 * t3 * t4
            z = t1 * t2 * t4 - t0 * t3 * t5

            return [w, x, y, z]

    def set_drone_attitude(self, roll_angle = 0, pitch_angle = 0, roll_rate = 0, pitch_rate = 0, yaw_rate = 0, thrust = .5):
        # Thrust = .5 holds the altitude, < .5 descends, >.5 ascends
        vel = self.get_drone_velocity()
        if (vel > 0 and vel < 2): # m/s
            thrust = .60
        elif (vel < 0 and vel >-2): # m/s
            thrust = .40
        else:
            thrust = .50

        # Create the mavlink message to send to the drone:
        msg = self.vehicle.message_factory.set_attitude_target_encode(0, # time - boot ms
                                                                      1, # Target system
                                                                      1, # Target COmponent
                                                                      0b00000000, # Mask, bit 1 LSB
                                                                      self.to_quaterion(roll_angle,pitch_angle,0, rad = True),# change to quaterions for roll, pitch, yaw
                                                                      roll_rate, # Body roll rate
                                                                      pitch_rate, # Body pitch rate
                                                                      yaw_rate, # Must be in radians
                                                                      thrust) # .5 for hover
        self.vehicle.send_mavlink(msg)
        duration = self.duration
        time.sleep(duration)

    # return only z velocity if not all velocity is needed
    def get_drone_velocity(self, all_velocity=False):
        if all_velocity:
            return self.vehicle.velocity
        return self.vehicle.velocity[2]

    def get_drone_attitude(self):
        # roll, pitch, yaw, posZ
        return (self.vehicle.attitude.roll,self.vehicle.attitude.pitch,self.vehicle.attitude.yaw, self.vehicle.location.global_relative_frame.alt)
    
    def get_drone_position(self): # xyz
        x = self.vehicle.location.global_relative_frame.lat - self.initial_position[0] 
        y = self.vehicle.location.global_relative_frame.lon - self.initial_position[1] 
        z = self.vehicle.location.global_relative_frame.alt - self.initial_position[2] -.75
        
        return (x,y,z)

    # Ask the simulator to return attitude information (private)
    def pget_simulator_state_info(self):
        mag, act_num, duration = (0,0,0)
        
        # The action the RC controller sends is part of the state information.
        mbv, mba, act_num, duration = self.get_RC_action() # mbv, mba, act_num, duration
        if mba is not None:
            mag = np.max(np.abs(mba))
        else:
            mag = np.max(np.abs(mbv))
        
        orient = self.get_drone_attitude()
        vel = self.get_drone_velocity(all_velocity = True)
        pos = self.get_drone_position()
        
        # print([orient[0],orient[1],orient[2],vel,mag,duration, self.dt])
        print('Roll, Pitch, Pos, Velx,Vely,Velz, Mag, Duration, dt')
        print(orient[0],orient[1],pos[2],vel[0],vel[1],vel[2],mag,duration, self.dt)
        
        state = np.array([orient[0],orient[1],pos[2],vel[2],mag,duration, self.dt]) # duration is proposed time, t_elapsed = time that has elasped since last state call
        encodedAction = self.lb.transform([act_num]).reshape(-1) # encode action
        extra_metadata = (pos[0],pos[1],vel[0],vel[1])
        
        # unpack and repack all state information into our state input
        self.current_state = np.concatenate((state,encodedAction))        
        #print('STATE: ', state)
        return self.current_state, extra_metadata

    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):

        # No action on our part...allow the RC action to pass through
        intervened = False
        self.dt = time.time() - self.prev_time # time between start of last action and the current action being taken
        self.prev_time = time.time()

        if action is 'No_Intervention':
            encoded_action = self.current_state[self.num_state_variables:] #obtain last rc command sent into NN by drone operator
            actionNum = np.asscalar(self.lb.inverse_transform(np.atleast_2d(encoded_action))) # decode back to action value
            print('Safe Action #: ', actionNum)

            # Either move by velocity, or move by angle at each timestep
            mbv,mba,_,_ = self.get_RC_action() # mbv, mba, action_number, duration
            duration = self.current_state[5] # duration logged from pget_simulator_state

            if mba is None: #  Move by Velocity
                quad_vel = self.get_drone_velocity(True)
                
                #self.set_drone_velocity(quad_vel[0]+mbv[0], quad_vel[1]+mbv[1], quad_vel[2]+mbv[2], duration)
                self.set_drone_velocity(quad_vel[0]+mbv[0], quad_vel[1]+mbv[1], quad_vel[2]+mbv[2], duration)
                time.sleep(0)

            else: # Move By Angle
                orient = self.get_drone_attitude() # get pitch, roll, yaw

                # pitch, roll, pos, yaw
                roll = orient[0] + duration * mba[0]
                pitch = orient[1] + duration * mba[1]
                self.set_drone_attitude(roll, pitch, mba[0], mba[1], duration)
                time.sleep(0)

        # We are going to intervene, since the neural net has predicted unsafe state-action
        else:
            print('Enter hover to save the drone')
#            self.vehicle.mode = VehicleMode("LOITER")
            self.set_drone_velocity(0,0,0,2)
            self.vehicle.armed = True
            intervened = True

        return intervened
    
    def movement_name(self, actionNum):
        dic = {0: 'No Accel', 1: 'Accel -X', 2: 'Accel -Y',3: 'Accel +Z',
               4: 'Accel +X', 5: 'Accel +Y', 6: 'Accel -Z', 7: 'Angle +X',
               8: 'Angle +Y', 9: 'Angle +Z', 10: 'Angle -X', 11: 'Angle -Y',
               12: 'Angle -Z'}
        return dic[actionNum]
    
    # Use this function to select a random movement or
    # if act_num is set to one of the functions action numbers, then all information about THAT action is returned
    # if act_num is not set, then the function will generate a random action
    def get_RC_action(self):
            
            self.count += 1
            if self.count < 10:
                rand_act_n = 6 # ENTER ACTION HERE
            else:
                rand_act_n = 3 # ENTER ACTION HERE
            if self.count > 21:
                self.count = 0
            move_by_vel = None
            move_by_angle = None
            
            self.scaling_factor = np.random.uniform(self.Lgain, self.Ugain)
            self.duration = .1

            if rand_act_n == 0: # no hover action...dont see the point right now
                move_by_vel = (0, 0, 0)
            elif rand_act_n == 1:
                move_by_vel = (self.scaling_factor, 0, 0)
            elif rand_act_n == 2:
                move_by_vel = (0, self.scaling_factor, 0)
            elif rand_act_n == 3:
                move_by_vel = (0, 0, -self.scaling_factor)
            elif rand_act_n == 4:
                move_by_vel = (-self.scaling_factor, 0, 0)
            elif rand_act_n == 5:
                move_by_vel = (0, -self.scaling_factor, 0)
            elif rand_act_n == 6:
                move_by_vel = (0, 0, self.scaling_factor)
            elif rand_act_n == 7:
                move_by_angle = (self.scaling_factor, 0, 0)
            elif rand_act_n == 8:
                move_by_angle = (0, self.scaling_factor, 0)
            elif rand_act_n == 9:
                move_by_angle = (0, 0, self.scaling_factor)
            elif rand_act_n == 10:
                move_by_angle = (-self.scaling_factor, 0, 0)
            elif rand_act_n == 11:
                move_by_angle = (0, -self.scaling_factor, 0)
            elif rand_act_n == 12:
                move_by_angle = (0, 0, -self.scaling_factor)
            # To make going up more likely
            elif rand_act_n == 13 or rand_act_n == 14:
                move_by_vel = (0, 0, -self.scaling_factor)
                rand_act_n = 6

            # Move by angle or move by velocity (one of the two will be set to None), meaning we either move by one or the other to generate a trajectory
            return move_by_vel, move_by_angle, rand_act_n, self.duration


    def reset(self):
        # connect to the dronekit API
        # print('Initializing Client')
        # self.vehicle = connect('udpin:0.0.0.0:14550', wait_ready = True)
        # print('Initialization Complete!')
        
        
#        """
#        Arms vehicle and fly to aTargetAltitude.
#        """
#        
#        print("Basic pre-arm checks")
#        # Don't try to arm until autopilot is ready
#        while not self.vehicle.is_armable:
#            print(" Waiting for vehicle to initialise...")
#            time.sleep(1)
#    
#        print("Arming motors")
#        # Copter should arm in GUIDED mode
#        self.vehicle.mode = VehicleMode("GUIDED")
#        self.vehicle.armed = True
#    
#        # Confirm vehicle armed before attempting to take off
#        while not self.vehicle.armed:
#            print(" Waiting for arming...")
#            time.sleep(1)
#        
#        time.sleep(5)
#        print("Taking off!")
#        self.vehicle.simple_takeoff(alt) # Take off to target altitude
#    
        # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
        #  after Vehicle.simple_takeoff will execute immediately).
        print("Initializing....")
        time.sleep(8)
#        self.initial_position = (self.vehicle.location.global_relative_frame.lat,self.vehicle.location.global_relative_frame.lon,self.vehicle.location.global_relative_frame.alt)
        self.initial_position = (0,0,self.vehicle.location.global_relative_frame.alt)
        print('Done Initializing!', 'Init PosZ:', self.initial_position)
        while True:
            print(" Altitude: ", self.vehicle.location.global_relative_frame.alt)
            #Break and return from function just below target altitude.
            if self.vehicle.location.global_relative_frame.alt>=self.target_alt*0.95:
                print("Reached target altitude")
                break
            time.sleep(1)
        
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        
        # Confirm vehicle armed before attempting to take off
#        while True:
#            print(" Waiting for arming...")
#            time.sleep(1)
#            
        print(" Altitude Ready:")
        time.sleep(2)
        print("Drone Model Start!")
        
        self.current_state, _ = self.pget_simulator_state_info()
        
        print('Z Init Vel: ', 0)
        print('Z Init Position: ', self.initial_position[2])
        
        self.prev_time = time.time()
        self.dt = 0

        return self.current_state, None

'''





