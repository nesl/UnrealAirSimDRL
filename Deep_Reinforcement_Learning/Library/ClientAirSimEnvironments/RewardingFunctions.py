# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:59:14 2018

@author: natsn
"""


import numpy as np

def drone_forest_racer_rewarding_function(collision_info, current_inertial_state, 
                                          max_altitude = 12, min_altitude = .45):
    mean_height = (max_altitude + min_altitude) / 2
    
    #print(collision_info.time_stamp, min_altitude, max_altitude, current_inertial_state)
    collided = False
    # PosX, PosY, PosZ, Vx, Vy, Vz, R, P, Y, Ax, Ay, Az, Rd, Pd, Yd, Rdd, Pdd, Ydd
    if  collision_info.has_collided:  # meters / second
        collided = True
        print("Collision Info: ", collision_info.has_collided)
        reward = -200
        done = True
        print('COLLIDED: ', collided)
        return reward, done
    
    # 2. Check for limits:
    # If we are above our z threshold, throw done flag
    if current_inertial_state[2] > max_altitude or current_inertial_state[2] < min_altitude:
        done = True
        reward = -50
        print('Out of Bounds, Resetting Episode!')
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
    #print(collision_info.time_stamp, min_altitude, max_altitude, current_inertial_state)
    
    collided = False
    if  collision_info.time_stamp or current_inertial_state[2] < min_altitude: # meters / second
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
