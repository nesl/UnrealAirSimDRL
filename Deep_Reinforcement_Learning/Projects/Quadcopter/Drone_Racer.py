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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disables debugging log

# Import Reinforcement learning Library
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Library")
from UnrealAirSimEnvironments import QuadcopterUnrealEnvironment
from CDQN import CDQN

# Import Local Helper Utils
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Util")
from ImageProcessing import trim_append_state_vector, fill_state_vector

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Some Globals for dimensioning and blocking
IMG_HEIGHT = 144
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_STEP = 4
IMG_VIEWS = 1 # Front Left, Front Center, Front Right

# Train the car to self drive -- SKEERTTTT!!!!!!!!!
def train_racing_car(primaryNetwork, targetNetwork, 
                     env, directory, 
                     NUM_EPISODES = 1000, EPISODE_ITERATIONS_MAX = 400,
                     COPY_PERIOD = 80, EPSILON = 1):
    
    for episode in range(NUM_EPISODES):
        print('Start racing! ', " Episode: ", episode)
        
        # Episode iterator, copy model iterator and flag signaling to copy parameters or restart episode
        episode_iterator = 0
        copy_model_iterator = 0
        COPY_MODEL_FLAG = False
        DONE_FLAG = False
        
        # Decay Epsilon to Greedy-Like
        ep = EPSILON / (episode_iterator + 1)
        if ep < .02:
            ep = .02
    
        # Copy model if we have reached copy_period rounds.
        if COPY_MODEL_FLAG:
            print('Copying model')
            targetNetwork.copy_from(primaryNetwork)
            copy_model_iterator = 0
            COPY_MODEL_FLAG = False
        
        # Reset states for the next training round
        obs4 = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_VIEWS)).reshape(1 ,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_VIEWS)
        obs4 = fill_state_vector(obs4, repeat = IMG_STEP - 1)
        
        # Returns inertial state vector, the image observation, and meta data (Time ellapsed)
        car_state, obs, meta =  env.reset()
        obs4 = trim_append_state_vector(obs4, obs)
        
        # Return and Loss
        episode_return = 0
        episode_loss = 0
        
        while not DONE_FLAG and episode_iterator < EPISODE_ITERATIONS_MAX:
            episode_iterator += 1
            
            # Sample action 
            action = primaryNetwork.sample_action(obs4, ep)
            
            # Update Stacked State
            prev_obs4 = obs4
            state, obs, reward, DONE_FLAG, meta = env.step(action) # new single observation
            
            obs4 = trim_append_state_vector(obs4, obs) # pop old and append new state obsv
            
            # Add new reward to running episode return
            episode_return += reward
            
            # Save new experience and train
            primaryNetwork.add_experience(prev_obs4, action, reward, obs4, DONE_FLAG)
            loss = primaryNetwork.train(targetNetwork) # train by using the training model
            
            if loss is not None: # Meaning, we have enough training exmaples to actually start training..
                episode_loss += loss
            
            # Increment Copy Model Iterator
            copy_model_iterator += 1
            
            # Signal a copy to the training model if we reach copy_period timesteps
            if copy_model_iterator % COPY_PERIOD == 0:
                COPY_MODEL_FLAG = True
            
            print("Episode: ", episode, ", Iteration: ", 
                  episode_iterator, ", Reward: ", reward, 
                  ", Loss: ", loss, ", Action: ", env.movement_name(action), ", Ellasped Time: ", meta)
            
            # Add Excel and GUI Listeners to Loop
        print("Total Episode Return: ", episode_return, ", Total Episode Loss: ", episode_loss)
        
def main():
    
    # Environment returns us the states, rewards, and communicates with the Unreal Simulator
    UREnv = QuadcopterUnrealEnvironment(mode = "both-rgb") # Returns both the image state (RGB) and the inertial state
    
    data_dir = "D:\\saved_NN_models\\TrainingModel\\safetyNet-276"
    model_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Car\\Data"
    
    xdims = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    n_outputs = 14 # Quadcopter control commands
    hidden_layer_sizes = [2000,1000]
    gamma = .9999
    learning_rate = 1e-3
    
    # Initialize the primary and target networks
    primaryNetwork = CDQN(xdims, n_outputs, hidden_layer_sizes, 
                 gamma, learning_rate = learning_rate)
    
    targetNetwork = CDQN(xdims, n_outputs, hidden_layer_sizes, 
                  gamma, learning_rate = learning_rate)
    
    # Instantiate Tensorflow Session
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = False
        #session = tf.Session(config=config)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    primaryNetwork.set_session(session)
    targetNetwork.set_session(session)
    
    # Restore Session if we have a previously trained model
    #primaryNetwork.restore_session(session)
    #targetNetwork.restore_session(session)
    
    # Train the Vehicle
    print('Network Ready!')
    train_racing_car(primaryNetwork, targetNetwork, UREnv, directory = model_dir)
    
    # Save the Session to the model directory
    print("Saving Session")
    primaryNetwork.save_session(model_dir)
    
    session.close()

if __name__ == '__main__':
    main()
