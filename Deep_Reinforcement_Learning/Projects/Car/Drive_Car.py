# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

# This script will take in states and actions from a simulated or real RC controller and determine whether the actions
# will lead to a failure...and if so will block the action from taking place.

# This algorithm uses an extended state space for training puposes

# Import Essentials
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
from XboxListener import XBoxListener
# Import Reinforcement learning Library
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\Library\\ClientAirSimEnvironments")
from ManualCarUnrealEnvironment import ManualCarUnrealEnvironment

# Train the car to self drive -- SKEERTTTT!!!!!!!!!
def train_racing_car(env, 
                     NUM_EPISODES = 1000):
    print("Xbox On!")
    xbl = XBoxListener(.025)
    xbl.init()
    for episode in range(NUM_EPISODES):
        print('Reset racing!', "Episode: ", episode)
        # Returns inertial state vector, the image observation, and meta data (Time ellapsed)
        env.reset()
        DONE_FLAG = False
        action = {'brake': 0, 'steering': 0, 'throttle': 0}
        while not DONE_FLAG:
            # Sample action 
            xbox_controls = xbl.get()
            if xbox_controls is None:
                _, _, _, DONE_FLAG, _ = env.step(action) # new single observation
            else:
                if xbox_controls["LA"] == "ly":
                    action['throttle'] = 2*xbox_controls["LAV"]
                elif xbox_controls["LA"] == "lx":
                    action['steering'] = 2*xbox_controls["LAV"]
                elif xbox_controls["LA"] == "rt":
                    action['brake'] = xbox_controls["LAV"]
                    # Update Stacked State
                    _, _, _, DONE_FLAG, _ = env.step(action) # new single observation
            
            
def main():
    
    vehicle_name = "Car1"
    input_mode = "xbox"
    image_mask_FC_FR_FL = [True, True, True] # Full front 180 view
    action_duration = .025
    sim_mode = "both_gray"
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_STEP = 3
    UREnv = ManualCarUnrealEnvironment(vehicle_name = vehicle_name,
                                        input_mode = input_mode,
                                        action_duration = action_duration,
                                        image_mask_FC_FR_FL = image_mask_FC_FR_FL,
                                        sim_mode = sim_mode,
                                        IMG_HEIGHT = IMG_HEIGHT,
                                        IMG_WIDTH = IMG_WIDTH,
                                        IMG_STEP = IMG_STEP)
    
    
    # Train the Vehicle
    print('Vehicle Ready!')
    train_racing_car(UREnv)
    
if __name__ == '__main__':
    main()
