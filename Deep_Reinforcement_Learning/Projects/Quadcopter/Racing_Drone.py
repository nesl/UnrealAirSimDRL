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
sys.path.append("../../Deep_Reinforcement_Learning/Library")
from UnrealAirSimEnvironments import QuadcopterUnrealEnvironment
from CDQN import CDQN
import AirSimGUI

# Import Local Helper Utils
sys.path.append("../../Util")
from ImageProcessing import trim_append_state_vector, fill_state_vector

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Some Globals for dimensioning and blocking
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1 # Gray or RGB or Depth (4-D)
IMG_STEP = 3
IMG_VIEWS = 3 # Front Left, Front Center, Front Right

# Train the car to self drive -- SKEERTTTT!!!!!!!!!
def train_racing_drone(primaryNetwork, targetNetwork, 
                     env, directory, 
                     NUM_EPISODES = 1000, EPISODE_ITERATIONS_MAX = 100,
                     COPY_PERIOD = 200, EPSILON = 1):
    
    # Set up GUI Video Feeder
    vehicle_names = ["drone1"]
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
        DONE_FLAG = False
    
        # Copy model if we have reached copy_period rounds.
        if COPY_MODEL_FLAG:
            print('Copying model')
            targetNetwork.copy_from(primaryNetwork)
            copy_model_iterator = 0
            COPY_MODEL_FLAG = False
            print('Copying model complete!')
            time.sleep(1)
        
        # Train the model more now between rounds so delay is less
        #primaryNetwork.train(targetNetwork, 20) # num times
        
        # Save Model Weights
        if (episode + 1) % 10 == 0:
            primaryNetwork.save_session(directory)
        
        
        # Reset s*tates for the next training round
        obs4 = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_STEP * IMG_VIEWS))
        
        # Returns inertial state vector, the image observation, and meta data (Time ellapsed)
        car_state, obs, meta =  env.reset()
        #print(obs.shape, obs4.shape)
        obs4 = trim_append_state_vector(obs4, obs, pop_index = IMG_VIEWS * IMG_CHANNELS)
        
        # Return and Loss
        episode_return = 0
        episode_loss = 0
        
        while not DONE_FLAG and episode_iterator < EPISODE_ITERATIONS_MAX:
            episode_iterator += 1
            tic = time.time()
            
            # Sample action 
            action = primaryNetwork.sample_action(obs4, ep)
            
            # Update Stacked State
            prev_obs4 = obs4
            if episode_iterator < 8:
                state, obs, reward, DONE_FLAG, meta = env.step(env.action_name("Vz")) # new single observation
            else:
                state, obs, reward, DONE_FLAG, meta = env.step(action) # new single observation
            
            # Pause Sim
            env.client.simPause(True)
            
            # Reduce dimentionality of obs
            obs4 = trim_append_state_vector(obs4, obs, pop_index = IMG_VIEWS * IMG_CHANNELS) # pop old and append new state obsv
            # Add new reward to running episode return
            episode_return += reward
            
            # Save new experience and train
            primaryNetwork.add_experience(prev_obs4, action, reward, obs4, DONE_FLAG)
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
            d1 = dict.fromkeys(vehicle_names, np.array(obs4 * 255, dtype = np.uint8))
            d2 = dict.fromkeys(vehicle_names, state)
            childImgConn.send(d1) # Make it a image format (unsigned integers)
            childStateConn.send(d2)
            print("GUI Process Update Time: ", time.time() - tic2)
            
            # Print Output
            #print(state)
            print("Episode: ", episode, ", Iteration: ", 
                  episode_iterator, ", Reward: ", reward,
                  ", Loss: ", loss, ", Action: ", env.action_name(action), ", Ellasped Time: ", time.time() - tic)
            env.client.simPause(False) # Play
            
            # Add Excel and GUI Listeners to Loop
        print("Total Episode Return: ", episode_return, ", Total Episode Loss: ", episode_loss)
        
def main():
    
    # Environment returns us the states, rewards, and communicates with the Unreal Simulator
    UREnv = QuadcopterUnrealEnvironment(image_mask_FC_FR_FL = [True, True, True], mode = "both_gray_normal") # Returns both the image obs and the inertial state
    
    #data_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Car\\Data"
    model_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Quadcopter\\Models2"
    
    xdims = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_VIEWS * IMG_STEP)
    n_outputs = 14 # Quad Commands -- See quad drive env
    hidden_layer_sizes = [800,400]
    gamma = .99
    learning_rate = 1e-2
    trainig_batch_size = 32
    
    # Initialize the primary and target networks
    primaryNetwork = CDQN(xdims, n_outputs, hidden_layer_sizes, 
                 gamma, learning_rate = learning_rate, batch_sz = trainig_batch_size )
    
    targetNetwork = CDQN(xdims, n_outputs, hidden_layer_sizes, 
                  gamma, learning_rate = learning_rate, batch_sz = trainig_batch_size)
    
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
    train_racing_drone(primaryNetwork, targetNetwork, UREnv, directory = model_dir)
    
    # Save the Session to the model directory
    print("Saving Session")
    primaryNetwork.save_session(model_dir)
    
    session.close()

if __name__ == '__main__':
    main()
