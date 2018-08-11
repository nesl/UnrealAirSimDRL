# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

# This script will take in states and actions from a simulated or real RC controller and determine whether the actions
# will lead to a failure...and if so will block the action from taking place.

# This algorithm uses an extended state space for training puposes

# Essentials
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Import Reinforcement learning Library
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Deep_Reinforcement_Learning\\Library")
from environments import CarUnrealEnvironment
from DQN import DQN
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Some Globals for dimensioning and blocking
IMG_HEIGHT = 144
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_STEP = 4
IMG_VIEWS = 1

# Train the car to self drive -- SKEERTTTT!!!!!!!!!
def train_racing_car(primaryNetwork, targetNetwork, env, directory, rounds, copy_period = 80):
    print('Start racing!')
    it = 0
    step = rounds[0]
    copy_model = False

    # Decaying Epsilon Greedy
    ep = primaryNetwork.epsilon / (it + 1)
    if ep < .02:
        ep = .02
    
    # Copy model if we have reached copy_period rounds.
    if copy_model:
        print('Copying model')
        targetNetwork.copy_from(primaryNetwork)
        it = 0
        copy_model = False

    # Reset Variables for next round
    obs4 = np.array(np.zeros(IMG_STEP, ))
    obs, _ =  env.reset()
    obs4 = primaryNetwork.state_stack(obs4,obs) # fills obs4 with same info 4 times
    done = False
    totalreward = 0
    totalroundloss = 0
    local_round = 0

    while not done and local_round < 300:
        local_round += 1

        print('Local round: ', local_round)

        # Action = 0: Do nothing, Action 1: intervene in flight progress
        action, actionNum = primaryNetwork.sample_action(obs4, ep)

        # Update stacked state
        prev_obs4 = obs4
        obs, reward, done, extra_metadata = env.step(action) # new single observation
        obs4 = primaryNetwork.state_stack(obs4,obs) # pop old and append new state obsv

        totalreward += reward

        primaryNetwork.add_experience(prev_obs4, actionNum, reward, obs4, done)
        loss = primaryNetwork.train(targetNetwork) # train by using the training model
        totalroundloss += loss
        
        # Increment Counter
        it += 1
        
        # Copy model to the TrainingModel if we reach copy_period timesteps
        if it % copy_period == 0:
            copy_model = True
        
        # Save total reward recieved for csv
        model.total_losses.append(totalroundloss / local_round)
        model.total_rewards.append(totalreward)

    
    return model.total_rewards, model.total_losses



def main():
    minimax_z_thresh = (.35, 65) # in absolute z-coords -- negative lower position enables crashes to ground

    env = QuadrotorSensorySafetyEnvironment(minimax_z_thresh)
    
    ####### CHANGE THESE TO SPECIFC REQUIREMENTS EACH TIME THE SCRIPT IS CALLED
    directory = "D:\\saved_NN_models\\TrainingModel\\safetyNet-276"
    LRound = 1
    URound = 301
    rounds = np.arange(LRound,URound)
    Lgain = .15 # .2,.3
    Ugain = .30 # .4,.5
    env.Lgain = Lgain
    env.Ugain = Ugain
    learning_rate = 1e-3
    #########################################################################
    
    time_steps = 4
    state_vars = env.total_state_variables # roll, pitch, Vz, PosZ, encoded_RC_action
    n_inputs = time_steps*state_vars
    n_outputs = 2 # Block Signal, Dont Block Signal

    sizes = [250,800,800,200]
    gamma = .9999
    
    model = DQN(n_inputs, n_outputs, sizes, gamma, learning_rate = learning_rate)
    tmodel = DQN(n_inputs, n_outputs, sizes, gamma, learning_rate = learning_rate)
    
    # Instantiate Tensorflow Session
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = False
    #session = tf.Session(config=config)
    session = tf.Session()
    # Set Sessions in model + tmodel with restore
    init = tf.global_variables_initializer()
    session.run(init)


##########################################################################################
# Uncomment if model should be restored!
   #Set Sessions in model + tmodel with restore
#    saver = tf.train.Saver()
#    #saver.restore(session, tf.train.latest_checkpoint(directory))
#    saver.restore(session, tf.train.latest_checkpoint(directory))
#    model.epsilon = .1
#    print('Safety Model Restored!')
###########################################################################################

    model.set_session(session)
    tmodel.set_session(session)

    # Play the Game N Times
    print('Safety Model Ready!')
    totalrewards, totallosses = train_safety_agent(model, tmodel, env, directory = directory, rounds = rounds )

    # Direct Output results
    plt.figure(1)
    plt.scatter(rounds,totalrewards)
    plt.title("Return Per round")
    plt.xlabel("Round")
    plt.ylabel("Score")

    plt.figure(2)
    run_avg = np.cumsum(totalrewards) / np.array(range(1,len(totalrewards)+1))
    plt.plot(rounds, run_avg)
    plt.title("Running Average Score vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Running Average Score")

    plt.figure(3)
    plt.plot(rounds,totallosses)
    plt.title("Loss vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()

    session.close()

if __name__ == '__main__':
    main()
