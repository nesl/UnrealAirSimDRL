# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:26:01 2018

@author: natsn
"""

import numpy as np
import time
from airsim import client
from airsim.types import Vector3r, Quaternionr
import AirSimGUI
import multiprocessing 
import threading
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\Util")
from ImageProcessing import trim_append_state_vector, fill_state_vector

# The drone is incentivized to use its vision to cruise around the world at 5 meters
# The faster the drone moves around the world, the more points it achieves


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





