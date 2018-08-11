# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

# This script will take in states and actions from a simulated or real RC controller and determine whether the actions
# will lead to a failure...and if so will block the action from taking place.

# This algorithm uses an extended state space for training puposes


import os
import sys
import numpy as np
import tensorflow as tf
import csv

sys.path.append("C:\\Users\\natsn\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Deep_Reinforcement_Learning\\Air_Sim\\Drone_Code\\")

from environments import QuadrotorSensoryDroneKitEnvironment
from quad_dqn import DQN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# plays one round of the environment you pass in
def test_dronekit_agent(model, tmodel, env, directory, flight = 0, date = ""):
    print('Begin Learning!')
    epsilon = 0
    
    # Keep track of data
    data_log = {'Round': [],'Local_Round': [],'Reward': [], 'Loss': [], 'Action': [], 'PosZ': [],
                'Vz': [], 'Pitch': [], 'Roll': [], 'RC_Action': [], 'Gain':[], 'Duration': [],
                'Elasped': [], 'Vx': [], 'Vy': [], 'Px': [], 'Py': []}
    
    
    # Reset Variables for next round
    obs4 = []
    obs, _ =  env.reset()
    obs4 = model.state_stack(obs4,obs) # fills obs4 with same info 4 times
    done = False
    local_round = 0

    while not done and local_round < 500:
        local_round += 1

        print('Local round: ', local_round)

        # Action = 0: Do nothing, Action 1: intervene in flight progress
        action, actionNum = model.sample_action(obs4, epsilon)

        # Update stacked state
        obs, done, extra_metadata = env.step(action) # new single observation
        obs4 = model.state_stack(obs4,obs) # pop old and append new state obsv


        # Append information for analysis
        print('Pos Z: ', obs[2])
        print('Vz: ', obs[3])
        print('Pitch: ', obs[1])
        print('Roll: ', obs[0])
        rc_act = np.asscalar(env.lb.inverse_transform(np.atleast_2d(obs[env.num_state_variables:])))
        print('RC Action:', env.movement_name(rc_act)) # Print the RC encoded action
        print('Gain: ', obs[4], 'm/s**2 or rad/s')
        print('Duration: ', obs[5], ' seconds')
        print('Elasped: ', obs[6], ' seconds')
        print('')
        
        data_log['Round'].append('TEST ROUND')
        data_log['Loss'].append('NULL')
        data_log['Local_Round'].append(local_round)
        data_log['Reward'].append('NULL')
        data_log['Action'].append(actionNum)
        data_log['PosZ'].append(obs[2])
        data_log['Vz'].append(obs[3])
        data_log['Pitch'].append(obs[1])
        data_log['Roll'].append(obs[0])
        data_log['Gain'].append(obs[4])
        data_log['RC_Action'].append(rc_act)
        data_log['Duration'].append(obs[5])
        data_log['Elasped'].append(obs[6])
        data_log['Px'].append(extra_metadata[0])
        data_log['Py'].append(extra_metadata[1])
        data_log['Vx'].append(extra_metadata[2])
        data_log['Vy'].append(extra_metadata[3])

    
    #print("Round:", 0, "total reward:", np.sum(model.total_rewards), "eps:", 0, "avg reward (last 25):", np.average(model.total_rewards[-25:]))
    print('Wrinting RL Data to File')
    file = directory + "RL_DroneKit_Test_Data" + "Flight" + str(flight) + "_" + date + "_" + ".csv"
    out = open(file, "a+", newline = '')
    wr = csv.writer(out,dialect = 'excel')
    wr.writerow(data_log.keys())
    wr.writerows(zip(*data_log.values()))
    out.close()
    
    
def main():
    minimax_z_thresh = (.35, 65) # in absolute z-coords -- negative lower position enables crashes to ground
    
    env = QuadrotorSensoryDroneKitEnvironment(minimax_z_thresh)
    
    ####### CHANGE THESE TO SPECIFC REQUIREMENTS EACH TIME THE SCRIPT IS CALLED
    directory = "D:\\saved_NN_models\\TrainingModel\\"
    date = "2_11_2018"
    flight = 5
    Lgain = 2 #.2,.3
    Ugain = 2 #.4,.5
    env.Lgain = Lgain
    env.Ugain = Ugain
    #########################################################################
    
    time_steps = 4
    state_vars = env.total_state_variables # roll, pitch, Vz, PosZ, encoded_RC_action
    n_inputs = time_steps*state_vars
    n_outputs = 2 # Block Signal, Dont Block Signal
    
    sizes = [250,800,800,200]
    gamma = .9999
    
    model = DQN(n_inputs, n_outputs, sizes, gamma)
    tmodel = DQN(n_inputs, n_outputs, sizes, gamma)
    
    # Instantiate Tensorflow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)
    
    # Set Sessions in model + tmodel with restore
    init = tf.global_variables_initializer()
    session.run(init)
    
    
##########################################################################################
# Uncomment if model should be restored!
    # Set Sessions in model + tmodel with restore
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint(directory))
    model.epsilon = 0
    print('Safety Model Restored!')
###########################################################################################

    model.set_session(session)
    tmodel.set_session(session)

    # Play N rounds of the game
    print('Safety Model Ready!')
    test_dronekit_agent(model, tmodel, env, directory, flight = flight, date = date)

    session.close()

if __name__ == '__main__':
    main()
