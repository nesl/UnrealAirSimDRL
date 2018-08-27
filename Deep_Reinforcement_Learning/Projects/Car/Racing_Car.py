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


# Import Reinforcement learning Library
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Library")
from CDQN import CDQN
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Library\\ClientAirSimEnvironments")
from AutoCarUnrealEnvironment import AutoCarUnrealEnvironment

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Train the car to self drive -- SKEERTTTT!!!!!!!!!
def train_racing_car(primaryNetwork, targetNetwork, 
                     env, directory, 
                     NUM_EPISODES = 1000, EPISODE_ITERATIONS_MAX = 200,
                     COPY_PERIOD = 50, EPSILON = 1):
    
    COPY_MODEL_FLAG = False
    copy_model_iterator = 0
    for episode in range(NUM_EPISODES):
        print('Reset racing!', "Episode: ", episode)
        env.client.simPause(True)
        
        # Decay Epsilon to Greedy-Like
        ep = EPSILON / np.sqrt(episode + 1)
        if ep < .02:
            ep = .02
        
        # Episode iterator, copy model iterator and flag signaling to copy parameters or restart episode
        episode_iterator = 0
        DONE_FLAG = False
        
        # Train the model more now between rounds so delay is less
        #primaryNetwork.train(targetNetwork, 20) # num times
        
        # Save Model Weights
        if (episode + 1) % 10 == 0:
            primaryNetwork.save_session(directory)
        
        # Returns inertial state vector, the image observation, and meta data (Time ellapsed)
        state, obs4, meta =  env.reset()
        
         # Copy model if we have reached copy_period rounds.
        if COPY_MODEL_FLAG:
            env.client.simPause(True)
            print('Copying model')
            targetNetwork.copy_from(primaryNetwork)
            copy_model_iterator = 0
            COPY_MODEL_FLAG = False
            print('Copying model complete!')
            env.client.simPause(False)
        
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
                state, obs4, reward, DONE_FLAG, meta = env.step(env.action_num("Throttle Up")) # new single observation
            else:
                state, obs4, reward, DONE_FLAG, meta = env.step(action) # new single observation
            
            # Pause Sim
            env.client.simPause(True)
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
            
            # Print Output
            print("Episode: ", episode, ", Iteration: ", 
                  episode_iterator, ", Reward: ", reward,
                  ", Loss: ", loss, ", Action: ", env.action_name(action), ", Ellasped Time: ", time.time() - tic)
            env.client.simPause(False)
            
            # Add Excel and GUI Listeners to Loop
        print("Total Episode Return: ", episode_return, ", Total Episode Loss: ", episode_loss)
        
def main():
    
    #data_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Projects\\Car\\Data"
    model_dir = "D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Network_Models\\Car\\model_racing_car"
    
    vehicle_name = "Car1"
    image_mask_FC_FR_FL = [True, True, True] # Full front 180 view
    action_duration = .08
    sim_mode = "both_gray"
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_STEP = 3
    UREnv = AutoCarUnrealEnvironment(vehicle_name = vehicle_name,
                                        action_duration = action_duration,
                                        image_mask_FC_FR_FL = image_mask_FC_FR_FL,
                                        sim_mode = sim_mode,
                                        IMG_HEIGHT = IMG_HEIGHT,
                                        IMG_WIDTH = IMG_WIDTH,
                                        IMG_STEP = IMG_STEP)
    
    
    # RL Agent
    IMG_CHANNELS = 1
    if 'rgb' in sim_mode:
        IMG_CHANNELS = 3
    IMG_VIEWS = np.sum(np.array(image_mask_FC_FR_FL, dtype = np.int))
    xdims = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS * IMG_VIEWS * IMG_STEP)
    n_outputs = 7 # Drive Commands -- See car drive env
    
    # Layer+Network Config
    hidden_layer_sizes = [800,400]
    gamma = .99
    learning_rate = 1e-2
    trainig_batch_size = 8
    
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
    train_racing_car(primaryNetwork, targetNetwork, UREnv, directory = model_dir)
    
    # Save the Session to the model directory
    print("Saving Session")
    primaryNetwork.save_session(model_dir)
    
    session.close()

if __name__ == '__main__':
    main()
