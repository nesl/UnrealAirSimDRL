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
import matplotlib.pyplot as plt
import csv

from environments import QuadrotorSensorySafetyEnvironment
from quad_dqn import PIDCutoff

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# plays one round of the environment you pass in
def pid_safety_agent(model, env, directory, rounds, extra_file_tag = ""):
    print('Begin Learning!')
    # Keep track of data
    data_log = {'Round': [],'Local_Round': [],'Reward': [], 'Loss': [], 'Action': [], 'PosZ': [],
                'Vz': [], 'Pitch': [], 'Roll': [], 'RC_Action': [], 'Gain':[], 'Duration': [],
                'Elasped': [], 'Vx': [], 'Vy': [], 'Px': [], 'Py': []}
    
    for i in rounds:
        # Decaying Epsilon Greedy

        print("Round ", i)

        # Reset Variables for next round
        obs, _ =  env.reset()
        done = False
        totalreward = 0
        local_round = 0

        while not done and local_round < 50:
            local_round += 1

            print('Local round: ', local_round)

            # Action = 0: Do nothing, Action 1: intervene in flight progress
            action, actionNum = model.sample_action(obs)

            # Update stacked state
            obs, reward, done, extra_metadata = env.step(action) # new single observation
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
            
            #model.add_experience(prev_obs, actionNum, reward, obs, done)
            
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

    print("Round:", i, "total reward:", np.sum(model.total_rewards), "avg reward (last 25):", np.average(model.total_rewards[-25:]))
    print('Wrinting RL Data to File')
    file = directory + "PID_Data" + extra_file_tag + str(rounds[0]) + "_" + str(rounds[-1]) + ".csv"
    
    out = open(file, "w+", newline = '')
    wr = csv.writer(out,dialect = 'excel')
    wr.writerow(data_log.keys())
    wr.writerows(zip(*data_log.values()))
    out.close()
    
    print("Average Total Reward (Last 20): ", np.average(model.total_rewards[-20:]))    
    return model.total_rewards



def main():
    minimax_z_thresh = (.35, 65) # in absolute z-coords -- negative lower position enables crashes to ground

    env = QuadrotorSensorySafetyEnvironment(minimax_z_thresh)
    
    ####### CHANGE THESE TO SPECIFC REQUIREMENTS EACH TIME THE SCRIPT IS CALLED
    directory = "D:\\saved_NN_models\\TrainingModel\\"
    extra_file_tag = "ROLLTEST"
    LRound = 1
    URound = 30
    rounds = np.arange(LRound,URound)
    Lgain = .15 # .2,.3
    Ugain = .30 # .4,.5
    env.Lgain = Lgain
    env.Ugain = Ugain
    ###########################################################################
    
    
    model = PIDCutoff(np.pi/4,np.pi/4,(2,-2))
    
    # Play the Game N Times
    print('Safety Model Ready!')
    totalrewards = pid_safety_agent(model, env, directory, rounds, extra_file_tag = extra_file_tag)

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

if __name__ == '__main__':
    main()
