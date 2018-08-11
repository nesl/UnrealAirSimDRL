3# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from read_sensor_data import DroneStatus
from subprocess import Popen
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from q_learning_bins import plot_running_avg


class environment:

    def __init__(self):
        
        # States will be pitch, roll, vx,vy,vz
        # Number of states = 5
        s = np.zeros(5)
        self.current_state = np.array(s)
        self.prev_state = np.array(s)
        
        # Create drone status member. Useful for rebooting drone
        self.drone_status = DroneStatus()
        
        # We want to input an action and have that affect the simulator
        # We want to return the next state, the reward, whether we have terminated from the simulator
    
        # This function will input an action command like OpenAI gym
        # The actions are as follows:
        # 0 Denotes that the system should continue to operate as it is...no safety risk
        # 1 Denotes that the agent believes the current state is not safe, and will cut off further action to the vehicle
        
    # The action will be to Go Left / Right / Up / Down for 50 ms
    def step(self, action):
        # If the try block fails, then this means we have crashed the drone
        # 0 = LEFT, 1 = RIGHT, 2 = UP, 3 = DOWN
        try:
            # 1. Take action in the simulator based on the DRL predict function
            self.do_action(action)
            #2. Save previous state 
            self.last_state = self.current_state
            #3. Return new state information from the simulator
            self.current_state = self.pget_simulator_state_info()
            #4. Collect reward from taking such action
            reward = self.calc_reward()
            done = False
            return (self.current_state,reward,done)
        
        # If we have ended in a crash state, reboot simulator and assign huge negative reward
        except:
            done = True
            self.current_state = self.last_state 
            reward = -200
            return (self.current_state,reward,done)
            
    # Ask the simulator to return attitude information (private)
    def pget_simulator_state_info(self):
        # this part needs to return the 10 raw data values of accelerom, gyro, mag, altitude
        pyr = self.drone_status.get_data()
        vel_xyz = self.drone_status.get_velocity() # Returns Vx, Vy, Vz
        return np.array([pyr[0],pyr[2],vel_xyz[0],vel_xyz[1],vel_xyz[2]]) 
        
    #Ask the simulator to perform control command or block control command
    def do_action(self,action): 
        # channel override to the drone
        # wait 5ms
        
    # The reward at every timestep will be around zero
    def calc_reward(self):
        # the reward will be around 1 for relatively low angles, and increasingly negative for higher angles
        #return 1 - ((3*self.current_state[0])**2 + (3*self.current_state[1])**2)
        return 1
        
    def reset(self):
        RESTART_FILE = 'PATH THAT I NEED TO FILL IN'
        self.process = Popen(RESTART_FILE, shell = True)
        self.current_state = np.array(np.zeros(5))
        self.prev_state = np.array(np.zeros(5))
        time.sleep(.1) # wait 100 milliseconds
        return self.get_simulator_state_info()
        

# a version of HiddenLayer that keeps track of params
class HiddenLayer:

    def __init__(self, size_of_input, size_of_layer, f= tf.nn.tanh):
        # Weights
        self.W = tf.Variable(tf.random_normal([size_of_input, size_of_layer]))

        # Weight Parameters stored for copy model
        self.params = [self.W]

        # Set bias term
        self.b = tf.Variable(tf.random_normal([size_of_layer]))

        # Bias Parameter stored for copy model
        self.params.append(self.b)

        # Set activation function
        self.f = f

    def forward(self, X):
        a = tf.matmul(X, self.W) + self.b
        return self.f(a) 


class DQN:
    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_sz=32):

        #Parameters for tuning
        self.epsilon = 1
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma

        #Input and Output layer dimentions
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # create the graph. Self.layers hold a list of each hidden layer
        self.layers = []
        input_size = self.n_inputs

        for layer_size in hidden_layer_sizes:
            layer = HiddenLayer(input_size, layer_size)
            self.layers.append(layer)
            input_size = layer_size

        # final layer
        layer = HiddenLayer(input_size, self.n_outputs, lambda x: x) # final layer does not take an activation function
        self.layers.append(layer)
                
        # collect params for copy model
        self.params = []
        for layer in self.layers:
            self.params += layer.params # concatenate all members into params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # None = it can vary
        self.G = tf.placeholder(tf.float32, shape=(batch_sz,), name='G') # a return of the given state
        self.actions = tf.placeholder(tf.int32, shape=(batch_sz,), name='actions') # a list of actions per each input state

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        self.selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, self.n_outputs),reduction_indices=[1])
        
        self.cost = tf.reduce_sum(tf.square(self.G - self.selected_action_value))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)
        # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
        #self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(self.cost)

        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.total_rewards = []

    # Instantiate the session from outside of the model
    def set_session(self, session):
        self.session = session

    # Copy the model to the copy model, which is used to run predictions on
    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # now run them all
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
            # don't do anything if we don't have enough experience
            return 

        # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False) # returns a list of positional indexes

        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        
        # With our SARS' fourple, based on our initial state, we will take the next state the action landed us in (s') and compute the maximum next state reward we can achieve
        # It is very important that we call the predict function on the target_network
        max_ns_reward = np.max(target_network.predict(next_states), axis=1)
        #max_ns_reward = target_network.predict(next_states) # Currently we only have 1 output guess
        
        # Targets, aka the current hypothesized return from a state, is what we iteratively train on.
        targets = [r + self.gamma*mnsr if not done else r for r, mnsr, done in zip(rewards, max_ns_reward, dones)]

        # Call the optimizer. Predict the loss from the current batch using the return as the target goal, with respect to THAT action the agent has chosen
        #self.session.run(self.train_op,feed_dict= {self.X: states, self.G: targets})
        self.session.run(self.train_op,feed_dict= {self.X: states, self.G: targets, self.actions: actions})
    
    # This function will run each and every time we take a step inside our model.
    def add_experience(self, s, a, r, s2, done):
        # If we have more expereince than our upper limit, we will pop the first element
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)

        # We will then append the new observation onto the top of the stack
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    # Use decaying epsilon greedy to converge onto our policy
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_outputs) # Choose a random action 0,1,2,3
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)) # returns the argument number of the highest return found from the Q-Net
            #return self.predict(x) # returns the argyment number of highest return
            
    # plays one round of the environment you pass in
    def play_game(self, TrainingModel, env, rounds = 1, copy_period = 50):
        
        for i in range(rounds):
            # Decaying Epsilon Greedy
            ep = self.epsilon / np.sqrt(i+1)
            
            # Print out some Statistics
            if i % 25 == 0:
                print("Round:", i, "total reward:", np.sum(self.total_rewards), "eps:", ep, "avg reward (last 100):", np.average(self.total_rewards[-50:]))
            else:
                print("Round ", i)
            
            # Reset Variables for next round
            obs = env.reset()
            done = False
            totalreward = 0
            it = 0
            
            while not done:
                action = self.sample_action(obs, ep)
                prev_obs = obs
                obs, reward, done, _ = env.step(action)
                
                totalreward += reward
                # Add large negative reward if the agent has lost the game
                if done:
                    reward = - 200
                else:
                # Add memory to memory replay dictionary and train model using the training model
                self.add_experience(prev_obs, action, reward, obs, done)
                self.train(TrainingModel) # train by using the training model
                
                # Increment Counter
                it += 1

                # Copy model to the TrainingModel if we reach 50 time steps
                if it % copy_period == 0:
                    TrainingModel.copy_from(self)
            
            # Save total reward recieved and return
            self.total_rewards.append(totalreward)

        # Return preformance score
        print("Last Round's Reward: ", self.total_rewards[-1])
        print("Average Reward: ", np.average(self.total_rewards))
        if rounds > 20:
            print("Last 20 Rounds Average: ", np.average(self.total_rewards[-20:]))
        return self.total_rewards


def main():
    env = environment()
    
    n_inputs = 5 # roll, pitch, Vx, Vy, Vz
    n_outputs = 4 # L/R/U/D
    sizes = [200,200]
    gamma = .9

    model = DQN(n_inputs, n_outputs, sizes, gamma)
    tmodel = DQN(n_inputs, n_outputs, sizes, gamma)

    # Instantiate Tensorflow Session
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    # Set Sessions in model + tmodel
    model.set_session(session)
    tmodel.set_session(session)

    # Play the Game N Times
    N = 10000
    totalrewards = model.play_game(tmodel, env, rounds = N)

    # Observe how well the model converged to optimal
    #for i in range(10):
        #watch_episode(model, env)

    plt.scatter([i for i in range(N)],totalrewards)
    plt.title("Rewards")
    plt.show()


if __name__ == '__main__':
  main()