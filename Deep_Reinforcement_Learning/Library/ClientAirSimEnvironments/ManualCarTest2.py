# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:26:15 2018

@author: natsn
"""


import sys, os
import numpy as np
import time
import multiprocessing 
import threading
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\..\\Util\\Virtual_IMU")
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..")
import AirSimGUI
import SO3Rotation
from airsim import client
from airsim.types import Vector3r
from ImageProcessing import trim_append_state_vector
from VehicleBase import AirSimVehicle



    
# The AirSim Environment Class
# vehicle name is Car 1,2,3...
# action duration = how long to do action
# VehicleCamerasModesAndOrientation 
# Coordinates are all the same
class ManualCarUnrealEnvironment(AirSimVehicle):

    def __init__(self, vehicle_name = "Car1",
                 action_duration = .08,
                 VehicleCamerasModesAndOrientations = {"front_center": [np.array([]), (0,0,0)],
                     "front_right": [np.array([]), (0,0,0)],
                     "front_left": [np.array([]), (0,0,0)],
                     "fpv": [np.array([]), (0,0,0)],
                     "back_center": [np.array([]), (0,0,0)]},
                 IMG_HEIGHT = 128,
                 IMG_WIDTH = 128,
                 IMG_STEP = 1): AirSimVehicle.__init__(False, VehicleCamerasModesAndOrientations)
        
        
        # Set Environment Return options
        # Set Drive Commands to zero initially
        self.action_duration = action_duration
        self._throttle = 0 # Used as a constant gain factor for the action throttle. 
        self._steering = 0 # Each action lasts this many seconds
        self._brake  = 0
        
        # Simulator Image setup
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_STEP = IMG_STEP
            
        self.obs4 = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS * self.IMG_STEP * self.IMG_VIEWS))
        # The number of INERTIAL state variables to keep track of
        self.count_inertial_state_variables = 19 # Posx, Posy, PosZ, Vx, Vy, Vz, Ax, Ay, Az, AngVelx, AngVely, AngVelz, AngAccx, AngAccy, AngAccz, quatx, quaty, quatz, quatw
        # Throttle up, throttle down, increase brake, decrease break, left_steer, right_steer, No action
        self.count_car_actions = 7
        # Initialize the current inertial state to zero
        self.current_inertial_state = np.array(np.zeros(self.count_inertial_state_variables))

        # Set up GUI Video Feeder
        self.gui_data = {'obs': None, 'state': None, 'meta': None}
        self.vehicle_name = vehicle_name
        GUIConn, self.simEnvDataConn = multiprocessing.Pipe()
        self.app = AirSimGUI.CarGUI(GUIConn, vehicle_names = [vehicle_name])
        
        self.timing = {"Time To Do Action": 0, "Time To Grab Images": 0, 
                       "Time To Grab States": 0, "Time To Calculate Reward": 0, "Time To Step": 0}
        self.extra_metadata = None

# Car actions are through manual input, either xbox controller or keyboard listener
    def step(self, action):
        tic = time.time()
        # 1. Take action in the simulator based on the agents action choice
        self.do_action(action)
        # 2: Update the environment state variables after the action takes place
        self.pset_simulator_state_info() 
        
        # 3: Check Collision and Get timings
        self.timing["Time to Step"] = time.time() - tic
        collision = self.client.simGetCollisionInfo()
        done = collision.has_collided
        # Send the data collected off to gui
        self.send_to_gui(action, done)
        # return states, images, crash
        return self.current_inertial_state, self.obs4, self, done
        
    def send_to_gui(self, action, done):
        self.extra_metadata = {'action': 0, 'action_name': 0, 'env_state': 0,
                               'mode': self.mode, 'reward': 0, 'done': done, 'times': self.timing}
        self.gui_data['state'] = self.current_inertial_state
        self.gui_data['obs'] = self.obs4
        self.gui_data['meta'] = self.extra_metadata
        
        t_gui = threading.Thread(target = self.simEnvDataConn.send, args = ({self.vehicle_name : self.gui_data},))
        t_gui.start()
        t_gui.join()
    
    # Ask the simulator to return attitude information (private)
    def pset_simulator_state_info(self):
        # This function will set the latest simulator information within the class
        # The State's components will be 
        # Get Base Inertial States
        tic = time.time()
        state = self.client.simGetGroundTruthKinematics()
        pos = (state['position']['x_val'],state['position']['y_val'],state['position']['z_val'])
        vel = (state['linear_velocity']['x_val'],state['linear_velocity']['y_val'],state['linear_velocity']['z_val'])
        acc = (state['linear_acceleration']['x_val'],state['linear_acceleration']['y_val'],state['linear_acceleration']['z_val'])
        angVel = (state['angular_velocity']['x_val'],state['angular_velocity']['y_val'],state['angular_velocity']['z_val'])
        angAcc = (state['angular_acceleration']['x_val'],state['angular_acceleration']['y_val'],state['angular_acceleration']['z_val'])
        orien = (state['orientation']['x_val'],state['orientation']['y_val'],state['orientation']['z_val'], state['orientation']['w_val'])
 
        # Store the current state
        self.current_inertial_state = np.array([pos[0] - self.initial_position[0], # Pos
                                                pos[1] - self.initial_position[1], # 
                                                pos[2] - self.initial_position[2],
                                                vel[0] - self.initial_velocity[0], # Vel
                                                vel[1] - self.initial_velocity[1],
                                                vel[2] - self.initial_velocity[2],
                                                acc[0], acc[1], acc[2], # Acc
                                                angVel[0], angVel[1], angVel[2], #acc
                                                angAcc[0], angAcc[1], angAcc[2], #ang
                                                orien[0], orien[1], orien[2], orien[3]]) # Quat
    
    
    
        self.timing["Time To Grab States"] = time.time() - tic
        tic = time.time()
        self.get_new_images()
        self.timing["Time To Grab Images"] = time.time() - tic
        
    def get_new_images(self):
        # Construct the Image Dictionary 
        self.client.simPause(True)
        self.obs4 = self.images_retrieval()
        self.client.simPause(False)
        
    
    #Ask the simulator to perform control command or to block control command
    def do_action(self, action):
        tic = time.time()
        #print("thr " + throttle_cmd + "Bra " + breaking_cmd + "str " + steering_cmd )
        self.brake = action['brake']
        self.steering = action['steering']
        self.throttle = action['throttle']

        car_controls = client.CarControls()
        car_controls.throttle = self.throttle
        car_controls.steering = self.steering
        car_controls.brake = self.brake
        self.client.setCarControls(car_controls) # Send off to the simulator!
        time.sleep(self.action_duration)
        self.timing["Time To Do Action"] = time.time() - tic
    
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
        if (b <= 1 and b >= 0):
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
    
    def reset(self):
        # Reset the Car
        print('Reseting Car')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
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
        # States Images Done
        return state, self.images_retrieval(), False






