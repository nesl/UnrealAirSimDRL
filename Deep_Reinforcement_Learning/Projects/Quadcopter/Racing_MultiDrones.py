# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

# This script will take in states and actions from a simulated or real RC controller and determine whether the actions
# will lead to a failure...and if so will block the action from taking place.

# This algorithm uses an extended state space for training puposes

# Import Essentials
import os
import numpy as np
import tensorflow as tf
import sys
import time
import multiprocessing as mp
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disables debugging log

# Import Reinforcement learning Library
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\Library")
from UnrealAirSimEnvironments import MultiQuadcoptersUnrealEnvironment
from MultiCDQN import MasterCDQN
import CDQN
import AirSimGUI

# Import Local Helper Utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\..\\Util")
from ImageProcessing import trim_append_state_vector, fill_state_vector

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Some Globals for dimensioning and blocking
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1 # Gray or RGB or Depth (4-D)
IMG_STEP = 3
IMG_VIEWS = 3 # Front Left, Front Center, Front Right

# Train the drone to self fly -- SKEERTTTT!!!!!!!!!
def train_racing_drones(_vehicle_names, primaryNetwork, targetNetwork, 
                     env, directory, 
                     NUM_EPISODES = 1000, EPISODE_ITERATIONS_MAX = 100,
                     COPY_PERIOD = 200, EPSILON = 1):
    
    # Set up GUI Video Feeder
    vehicle_names = _vehicle_names
    parentImgConn, childImgConn = mp.Pipe()
    parentStateConn, childStateConn = mp.Pipe()
    app = AirSimGUI.QuadcopterGUI(parentStateConn, parentImgConn, vehicle_names = vehicle_names,
                                               num_video_feeds = IMG_VIEWS*IMG_STEP, isRGB = False)
    
    COPY_MODEL_FLAG = False
    copy_model_iterator = 0
    for episode in range(NUM_EPISODES):
        print('Reset racing!', "Episode: ", episode)
        
        # Decay Epsilon to Greedy-Like
        ep = EPSILON / np.sqrt(episode + 1)
        if ep < .02:
            ep = .02
        
        # Episode iterator, copy model iterator and flag signaling to copy parameters or restart episode
        episode_iterator = 0
        DONE_FLAGs = dict.fromkeys(vehicle_names,False)
        
        env.client.simPause(True)
        # Copy model if we have reached copy_period rounds.
        if COPY_MODEL_FLAG:
            print('Copying model')
            targetNetwork.copy_from(primaryNetwork.CDQN)
            copy_model_iterator = 0
            COPY_MODEL_FLAG = False
            print('Copying model complete!')
            time.sleep(1)
        env.client.simPause(False)
        
        # Train the model more now between rounds so delay is less
        #primaryNetwork.train(targetNetwork, 20) # num times
        
        # Save Model Weights
        if (episode + 1) % 10 == 0:
            primaryNetwork.save_session(directory)
        
        
        # Reset s*tates for the next training round
        obs4s = dict.fromkeys(vehicle_names, np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_STEP * IMG_VIEWS)))
        
        # Returns inertial state vector, the image observation, and meta data (Time ellapsed)
        states, obss, metas = env.reset()
        #print(obs.shape, obs4.shape)
        for vn in vehicle_names:
            obs4s[vn] = trim_append_state_vector(obs4s[vn], obss[vn], pop_index = IMG_VIEWS * IMG_CHANNELS)
        
        # Return and Loss
        episode_return = 0
        episode_loss = 0

        while not np.sum(np.array(list(DONE_FLAGs.values()), dtype = np.int)) and episode_iterator < EPISODE_ITERATIONS_MAX:
            episode_iterator += 1
            tic = time.time()
            
            # Sample quad actions
            actions = primaryNetwork.sample_actions(obs4s, ep)
            
            # Update Stacked States
            prev_obs4s = obs4s
            states, obss, rewards, DONE_FLAGs, metas = env.steps(actions) # new single observation
            
            # Pause Sim
            env.client.simPause(True)
            
            # Stack New Obsvs
            for vn in vehicle_names:
                obs4s[vn] = trim_append_state_vector(obs4s[vn], obss[vn], pop_index = IMG_VIEWS * IMG_CHANNELS) # pop old and append new state obsv
            
            # Add new rewards to running episode return
            episode_return += np.sum(np.array([rewards[vn] for vn in vehicle_names]))
            
            # Save new experience and train
            primaryNetwork.add_experiences(prev_obs4s, actions, rewards, obs4s, DONE_FLAGs)
            loss = primaryNetwork.train(targetNetwork) # train by using the training model
            
            # Append Loss
            if loss is not None: # Meaning, we have enough training exmaples to actually start training..
                episode_loss += loss
            
            # Increment Copy Model Iterator
            copy_model_iterator += 1
            
            # Signal a copy to the training model if we reach copy_period timesteps
            if copy_model_iterator % COPY_PERIOD == 0:
                COPY_MODEL_FLAG = True

            # Send off to the GUI Process!
            tic2 = time.time()
            gui_obs = dict.fromkeys(vehicle_names)
            for vn in vehicle_names:
                gui_obs[vn] = np.array(obs4s[vn]*255,dtype = np.uint8)
            childImgConn.send(gui_obs) # Make it a image format (unsigned integers)
            childStateConn.send(states)
            print("GUI Process Update Time: ", time.time() - tic2)
            
            # Print Output
            #print(rewards,actions,DONE_FLAGs)
            print("Episode: ", episode, ", Iteration: ", 
                  episode_iterator, ", Rewards: ", rewards[vehicle_names[0]],
                  ", Loss: ", loss, ", Action Dron 0: ", actions[vehicle_names[0]], ", Ellasped Time: ", time.time() - tic)
            env.client.simPause(False) # Play
            
            # Add Excel and GUI Listeners to Loop
        print("Total Episode Return: ", episode_return, ", Total Episode Loss: ", episode_loss)
        
def main():
    # Define Vehicles: Should be same as your json
    vehicle_names = ["Drone1", "Drone2"]
    min_altitude = .25
    max_altitude = 20
    # Environment returns us the states, rewards, and communicates with the Unreal Simulator
    UREnv = MultiQuadcoptersUnrealEnvironment(vehicle_names, 
                                              image_mask_FC_FR_FL = [True, True, True], 
                                              mode = "both_gray_normal", 
                                              max_altitude = max_altitude,
                                              min_altitude = min_altitude) # Returns both the env image obs and vehicle state depending on mode
    
    #data_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Car\\Data"
    model_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Quadcopter\\Models2"
    
    xdims = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_VIEWS * IMG_STEP)
    n_outputs = 14 # Quad Commands -- See quad drive env
    hidden_layer_sizes = [800,400]
    gamma = .99
    learning_rate = 1e-2
    trainig_batch_size = 32
    Max_Experiences = 8000
    Min_Experiences = 200
    
    # Initialize the primary and target networks
    primaryNetwork = MasterCDQN(vehicle_names, xdims, n_outputs, 
                                hidden_layer_sizes, gamma,
                                max_experiences = Max_Experiences,
                                min_experiences = Min_Experiences,
                                batch_size = trainig_batch_size, 
                                learning_rate = learning_rate)
    
    targetNetwork = CDQN.CDQN(xdims, n_outputs, hidden_layer_sizes, 
                  gamma, batch_sz = trainig_batch_size, learning_rate = learning_rate)
    
    # Instantiate Tensorflow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    
    primaryNetwork.set_session(session)
    targetNetwork.set_session(session)
    
    # Restore Session if we have a previously trained model
    #primaryNetwork.restore_session(model_dir)
    #targetNetwork.restore_session(model_dir)
    
    # Train the Vehicle
    print('Network Ready!')
    train_racing_drones(vehicle_names, primaryNetwork, targetNetwork, UREnv, directory = model_dir)
    
    # Save the Session to the model directory
    print("Saving Session")
    primaryNetwork.save_session(model_dir)
    
    session.close()

if __name__ == '__main__':
    main()
