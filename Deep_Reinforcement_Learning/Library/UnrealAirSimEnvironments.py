# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:26:01 2018

@author: natsn
"""

import numpy as np
import time
from airsim import client




# The drone is incentivized to use its vision to cruise around the world at 5 meters
# The faster the drone moves around the world, the more points it achieves

def drone_forest_racer_rewarding_function(collision_info, current_inertial_state, 
                                          max_altitude = 8, min_altitude = 2):
    mean_height = (max_altitude + min_altitude) / 2
    # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd
    collided = False
    if  collision_info.time_stamp:  # meters / second
        collided = True
        
    # 2. Check for limits:
    # If we are above our z threshold, throw done flag
    if current_inertial_state[2] > max_altitude or current_inertial_state[2] < min_altitude:
        done = True
        reward = -25
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
        reward_speed = current_inertial_state[3]**2 + current_inertial_state[4]**2
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
def car_racing_rewarding_function(collision_info, current_inertial_state, bias = 10):
# 1) Determine if a collision has occured

    collided = False
    reward = 0
    if  collision_info.time_stamp or collision_info.has_collided: # meters / second
        collided = True

    # Check for collision with ground
    if collided:
        reward = -100
        done = True
        print('COLLIDED! ', collided)
        return reward, done

    else:
        reward = current_inertial_state[4]**2 + current_inertial_state[5]**2 - bias
        done = False
        return reward, done

# The AirSim Environment Class for Quadrotor Safety
class QuadcopterUnrealEnvironment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    def __init__(self, max_altitude = 50,
                 min_altitude = .20,
                 image_mask_FC_FR_FL = [True, False, False], # Front Center, Front right, front left
                 reward_function = drone_forest_racer_rewarding_function,
                 mode = "both-rgb"):
        
        self.reward_function = reward_function
        self.mode = mode
        
        self.scaling_factor = .200 # Used as a constant gain factor for the action throttle. 
        self.action_duration = .150 # Each action lasts this many seconds
        
        # Gains on the control commands sent to the quadcopter #
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 15 # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd
        self.count_drone_actions = 14 # 6 Linear, 6 angular, 1 hover, 1 No Op (Dont change anything)
        
        # Initialize the current inertial state
        self.current_inertial_state = np.array(np.zeros(self.count_inertial_state_variables))
        
        # Initialize the IMAGE variables -- We Take in Front Center, Right, Left
        self.images_rgb = None
        self.images_rgba = None
        self.image_mask_rgb = np.array([ [0+3*i,1+3*i,2+3*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        self.image_mask_rgba = np.array([ [0+4*i,1+4*i,2+4*i] for m, i in zip(image_mask_FC_FR_FL, range(3)) if m]).reshape(-1)
        
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
        elif self.mode == "image_rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "image_rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "both-rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "both-rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], reward, done, self.extra_metadata)
        else:
            print("Invalid Mode!")
            
    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be Px, Py, Pz, Vx, Vy, Vz ,Ax, Ay, Az, Roll, Pitch, Yaw, Roll_d, Pitch_d, Yaw_d, AngAccX, AngAccY, AngAccZ
        
        # Get Base Inertial States
        MRS = self.client.getMultirotorState()
        pos = MRS.kinematics_estimated.position
        vel = MRS.kinematics_estimated.linear_velocity
        acc = MRS.kinematics_estimated.linear_acceleration
        pry = MRS.kinematics_estimated.orientation
        angVel = MRS.kinematics_estimated.angular_velocity
        angAcc = MRS.kinematics_estimated.angular_acceleration
        
        # Find ellapsed time
        self.toc = time.time()
        self.dt = self.toc - self.tic
        
        self.current_inertial_state = np.array([pos.x_val, pos.y_val, -1*pos.z_val,
                                                vel.x_val, vel.y_val, vel.z_val,
                                                acc.x_val, acc.y_val, acc.z_val,
                                                pry.x_val, pry.y_val, pry.z_val,
                                                angVel.x_val, angVel.y_val, angVel.z_val,
                                                angAcc.x_val, angAcc.y_val, angAcc.z_val])
        
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
        images = self.client.simGetImages([
            client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
            client.ImageRequest("1",client.ImageType.Scene, False, False), # Front Right
            client.ImageRequest("2", client.ImageType.Scene, False, False)]) # Front Left
        
        
        # Convert Images to RGBA and RGB Values. Store in Appropriate containers
        img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
        img_rgba_FC = img1d_FC.reshape(images[0].height, images[0].width, 4)
        img_rgb_FC = img_rgba_FC[:,:,0:3]

        img1d_FR = np.fromstring(images[1].image_data_uint8, dtype=np.uint8) 
        img_rgba_FR = img1d_FR.reshape(images[1].height, images[1].width, 4)
        img_rgb_FR = img_rgba_FR[:,:,0:3]

        img1d_FL = np.fromstring(images[2].image_data_uint8, dtype=np.uint8) 
        img_rgba_FL = img1d_FL.reshape(images[2].height, images[2].width, 4)
        img_rgb_FL = img_rgba_FL[:,:,0:3]
        
        # Can either use the RGBA images or the RGB Images
        self.images_rgba = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
        self.images_rgb = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
        
        # Store X and Y position information as well
        self.extra_metadata = self.dt # Can change this to return whatever
        
        # Start the timer again as the next state is saved. Return the current state and meta-data 
        self.tic = time.time()
        
    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        
        # Either move by velocity, or move by angle at each timestep
        mbv, mba, mbh, _ = self.get_RC_action(act_num = action)
        quadState = self.client.getMultirotorState()
        if mbv is not None: #  Move By Velocity
            quad_vel = quadState.kinematics_estimated.linear_velocity
            self.client.moveByVelocityAsync(quad_vel.x_val + mbv[0], quad_vel.y_val + mbv[1], quad_vel.z_val + mbv[2], self.action_duration)
            time.sleep(self.action_duration)
            
        elif mba is not None: # Move By Angle
            quad_pry = quadState.kinematics_estimated.orientation
            self.client.moveByAngleZAsync(quad_pry.x_val + mba[0], quad_pry.y_val + mba[1], quad_pry.w_val, quad_pry.z_val + mba[2], self.action_duration)
            time.sleep(self.action_duration)
        
        elif mbh is not None: # Move By Hover
            self.client.hoverAsync()   
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

    def movement_name(self, actionNum):
        dic = {0: 'No Action', 1: 'Hover', 2: 'Vx',3: 'Vy',
               4: 'Vz', 5: '-Vx', 6: '-Vy', 7: '-Vz',
               8: '+Roll', 9: '+Pitch', 10: '+Yaw', 11: '-Roll',
               12: '-Pitch', 13: '-Yaw'}
        return dic[actionNum]
    
    def reset(self):
        # Reset the Copter
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Quickly raise the quadcopter to a few meters -- Speed up
        for i in range(18):
            state = self.client.getMultirotorState()
            quad_vel = state.kinematics_estimated.linear_velocity
            quad_offset = (0, 0, -self.scaling_factor)
            self.client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], self.action_duration)
            time.sleep(.1)
        
        
        state = self.client.getMultirotorState()
        lin_vel = state.kinematics_estimated.linear_velocity
        xyz = state.kinematics_estimated.position
        
        print('Z Init Vel: ', lin_vel.z_val)
        print('Z Init Position: ', xyz.z_val)
        
        self.initial_position = (xyz.x_val, xyz.y_val, xyz.z_val)
        print('Reset Complete')
        
        # Start Timer for episode's first step:
        self.dt = 0
        self.tic = time.time()
        time.sleep(.5)
        
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, self.extra_metadata)
        elif self.mode == "image_rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "image_rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "both-rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "both-rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], self.extra_metadata)
        else:
            print("Invalid Mode!")
        



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
                 reward_function = car_racing_rewarding_function, mode = "image_rgb"):
        
        # Set reward function
        self.reward_function = reward_function
        
        # Set Environment Return options
        self.mode = mode
        self.mode
        
        # Set Drive Commands to zero initially
        self._throttle = 1 # Used as a constant gain factor for the action throttle. 
        self._steering = 0 # Each action lasts this many seconds
        self._brake  = 0
        
        self.THROTTLE_INC = .10
        self.THROTTLE_DEC = -.10
        self.BRAKE_INC = .10
        self.BRAKE_DEC = -.10
        self.STEER_LEFT_INC = -.05
        self.STEER_RIGHT_INC = .05
        
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
        

        # Connect to the AirSim simulator and begin:
        print('Initializing Car Client')
        self.client = client.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
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
    
    def step(self, action):
        
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Calculate reward
        reward, done = self.calc_reward()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, reward, done, self.extra_metadata)
        elif self.mode == "image_rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "image_rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], reward, done, self.extra_metadata)
        elif self.mode == "both-rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], reward, done, self.extra_metadata)
        elif self.mode == "both-rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], reward, done, self.extra_metadata)
        else:
            print("Invalid Mode!")
            
    # Ask the simulator to return attitude information (private)
    # Modes are: Return inertial information, return camera images, or you can return both
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        
        # Get Base Inertial States
        state = self.client.getCarState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        acc = state.kinematics_estimated.linear_acceleration
        angVel = state.kinematics_estimated.angular_velocity
        angAcc = state.kinematics_estimated.angular_acceleration
        
        # Collect and Display Elapsed Time Between States
        self.toc = time.time()
        self.dt = self.toc - self.tic
        
        # Store the current state
        self.current_inertial_state = np.array([pos.x_val, pos.y_val, pos.z_val*-1,
                                                vel.x_val, vel.y_val, vel.z_val,
                                                acc.x_val, acc.y_val, acc.z_val,
                                                angVel.x_val, angVel.y_val, angVel.z_val,
                                                angAcc.x_val, angAcc.y_val, angAcc.z_val])
        
        # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz 
        
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
        images = self.client.simGetImages([
            client.ImageRequest("0", client.ImageType.Scene, False, False), # Front Center
            client.ImageRequest("1", client.ImageType.Scene, False, False), # Front Right
            client.ImageRequest("2", client.ImageType.Scene, False, False)]) # Front Left
        
    
        # Convert Images to RGBA and RGB Values. Store in Appropriate containers
        img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
        img_rgba_FC = img1d_FC.reshape(images[0].height, images[0].width, 4)
        img_rgb_FC = img_rgba_FC[:,:,0:3]

        img1d_FR = np.fromstring(images[1].image_data_uint8, dtype=np.uint8) 
        img_rgba_FR = img1d_FR.reshape(images[1].height, images[1].width, 4)
        img_rgb_FR = img_rgba_FR[:,:,0:3]

        img1d_FL = np.fromstring(images[2].image_data_uint8, dtype=np.uint8) 
        img_rgba_FL = img1d_FL.reshape(images[2].height, images[2].width, 4)
        img_rgb_FL = img_rgba_FL[:,:,0:3]
        
        # Can either use the RGBA images or the RGB Images
        self.images_rgba = np.dstack((img_rgba_FC,img_rgba_FR,img_rgba_FL))
        self.images_rgb = np.dstack((img_rgb_FC,img_rgb_FR,img_rgb_FL))
        
        # Store X and Y position information as well
        self.extra_metadata = self.dt
        
        # Start the timer again as the next state is saved. Return the current state and meta-data 
        self.tic = time.time()
        
    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        
        # Returns the increment or decrement (AKA the DQN's Action choice)
        throttle_cmd, breaking_cmd, steering_cmd, _ = self.get_RC_action(act_num = action)
        
        #print("thr " + throttle_cmd + "Bra " + breaking_cmd + "str " + steering_cmd )
        
        self.throttle += throttle_cmd
        self.brake += breaking_cmd
        self.steering += steering_cmd
        
        car_controls = client.CarControls()
        car_controls.throttle = self.throttle
        car_controls.steering = self.steering
        car_controls.brake = self.brake
        self.client.setCarControls(car_controls) # Send off to the simulator!
            
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
    @throttle.setter
    def throttle(self, t):
        if t <= 1 and t >= 0:
            self._throttle = t
            self._brake = 0
        elif t < 0:
            print("Throttle Value too low")
            self._throttle = 0
        else: # >1 
            print("Throttle Value too high")
            self._throttle = 1
            self._brake = 0
    @brake.setter
    def brake(self, b):
        if b <= 1 and b >= 0:
            self._brake = b
            #self._throttle = 0
        elif b < 0:
            print("Break Value too low")
            self._break = 0
        else: #b>1
            print("Break Value too high")
            self._break = 1
            #self._throttle = 0
    @steering.setter
    def steering(self, s):
        if s <= 1 and s >= -1:
            self._steering = s
        elif s < -1:
            print("Steering Value too low")
            self._steering = -1
        else: #s>1
            print("Steering Value too high")
            self._steering = 1 
            
    def movement_name(self, actionNum):
        dic = {0: 'Throttle Up', 1: 'Throttle Down', 2: 'Steer Left',
               3: 'Steer Right', 4: 'Brake Up', 5: 'Brake Down', 6: 'No Action'}
        return dic[actionNum]
    
    def reset(self):
        # Reset the Copter
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        
        # Define initial Car position
        state = self.client.getCarState()
        xyz = (state.kinematics_estimated.position.x_val, 
               state.kinematics_estimated.position.y_val,
               state.kinematics_estimated.position.z_val)
        
        self.initial_position = (xyz[0], xyz[1], xyz[2])
        print('Reset Complete')
        
        # Start Timer for episode's first step:
        self.dt = 0
        self.tic = time.time()
        time.sleep(.5) # to make sure acceleration doesnt come back insane the first loop
        
        # Set the environment state and image properties from the simulator
        self.pset_simulator_state_info()
        
        if self.mode == "inertial":
            return (self.current_inertial_state, self.extra_metadata)
        elif self.mode == "image_rgb":
            return (self.images_rgb[:,:,self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "image_rgba":
            return (self.images_rgba[:,:,self.image_mask_rgba], self.extra_metadata)
        elif self.mode == "both-rgb":
            return (self.current_inertial_state, self.images_rgb[:,:, self.image_mask_rgb], self.extra_metadata)
        elif self.mode == "both-rgba":
            return (self.current_inertial_state, self.images_rgba[:,:, self.image_mask_rgba], self.extra_metadata)
        else:
            print("Invalid Mode!")
















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





