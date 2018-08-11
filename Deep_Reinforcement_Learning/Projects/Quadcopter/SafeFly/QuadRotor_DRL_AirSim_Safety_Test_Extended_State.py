# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

# This script will take in states and actions from a simulated or real RC controller and determine whether the actions
# will lead to a failure...and if so will block the action from taking place.

# This algorithm uses an extended state space for training puposes


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

from environments import QuadrotorSensorySafetyEnvironment
from quad_dqn import DQN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# plays one round of the environment you pass in
def test_safety_agent(model, tmodel, env, directory, rounds, copy_period = 80, file_name_extra = ""):
    print('Begin Learning!')
    step = 0
    epsilon = 0

    # Keep track of data
    data_log = {'Round': [],'Local_Round': [],'Reward': [], 'Loss': [], 'Action': [], 'PosZ': [],
                'Vz': [], 'Pitch': [], 'Roll': [], 'RC_Action': [], 'Gain':[], 'Duration': [],
                'Elasped': [], 'Vx': [], 'Vy': [], 'Px': [], 'Py': []}
    
    for i in rounds:
        step += 1
        # Reset Variables for next round
        obs4 = []
        obs, _ =  env.reset()
        obs4 = model.state_stack(obs4,obs) # fills obs4 with same info 4 times
        done = False
        totalreward = 0
        local_round = 0

        while not done and local_round < 200:
            local_round += 1

            print('Local round: ', local_round)

            # Action = 0: Do nothing, Action 1: intervene in flight progress
            action, actionNum = model.sample_action(obs4, epsilon)
            
            # Update stacked state
            obs, reward, done, extra_metadata = env.step(action) # new single observation
            obs4 = model.state_stack(obs4,obs) # pop old and append new state obsv

            totalreward += reward

            # Append information for analysis
            print('Reward: ', reward)
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
            
            data_log['Round'].append(i)
            data_log['Loss'].append('NULL')
            data_log['Local_Round'].append(local_round)
            data_log['Reward'].append(reward)
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

        # Save total reward recieved for csv
        model.total_rewards.append(totalreward)

        data_log['Round'].append(i)
        data_log['Action'].append('NULL')
        data_log['Loss'].append('NULL')
        data_log['Local_Round'].append('NULL')
        data_log['PosZ'].append('NULL')
        data_log['Vz'].append('NULL')
        data_log['Pitch'].append('NULL')
        data_log['Roll'].append('NULL')
        data_log['Reward'].append(totalreward)
        data_log['RC_Action'].append('NULL')
        data_log['Duration'].append('NULL')
        data_log['Elasped'].append('NULL')
        data_log['Gain'].append('NULL')
        data_log['Vx'].append('NULL')
        data_log['Vy'].append('NULL')
        data_log['Px'].append('NULL')
        data_log['Py'].append('NULL')
    
    # Return preformance score
    print("Last Round's Reward: ", model.total_rewards[-1])
    print("Average Reward: ", np.average(model.total_rewards))
    file = directory + "Safey_Test" + file_name_extra + ".csv"
    out = open(file, "w+", newline = '')
    wr = csv.writer(out,dialect = 'excel')
    wr.writerow(data_log.keys())
    wr.writerows(zip(*data_log.values()))
    out.close()
    return model.total_rewards
    
def main():
    minimax_z_thresh = (.1, 65) # in absolute z-coords -- negative lower position enables crashes to ground
    
    env = QuadrotorSensorySafetyEnvironment(minimax_z_thresh)
    
    ####### CHANGE THESE TO SPECIFC REQUIREMENTS EACH TIME THE SCRIPT IS CALLED
    directory = "D:\\saved_NN_models\\TrainingModel\\"
    file_name_extra = "RandomActions"
    LRound = 1
    URound = 50
    rounds = np.arange(LRound,URound)
    Lgain = .2 #.2,.3
    Ugain = .4 #.4,.5
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
    totalrewards = test_safety_agent(model, tmodel, env, directory = directory, rounds = rounds, file_name_extra = file_name_extra)

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
    plt.show()
    
    session.close()

if __name__ == '__main__':
    main()
