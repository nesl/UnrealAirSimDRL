#!/usr/bin/env python3
import sys, os
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/home/natsubuntu/catkin_ws/src/pibot_ai/scripts/Deep_Reinforcement_Learning/Library")
sys.path.append("/home/natsubuntu/catkin_ws/src/pibot_ai/scripts/Convolutional Neural Networks (CNN)")
sys.path.append("/home/natsubuntu/catkin_ws/src/pibot_ai/scripts/Neural_Network")
from FullyConnectedLayer import FullyConnectedLayer
from FullyConnectedResNetLayer import FullyConnectedResNetBlock
np.random.seed()
env = gym.make("MountainCarContinuous-v0")
print(env.reset())
print(env.action_space.sample())

# Fully Connected
class PolicyModel:
    def __init__(self, input_dim = 2, 
                output_dim = 1, hl_sizes = [64,64,64], 
                use_res_net = True, activation_fun = tf.nn.relu,
                training_rate = 1e-3, eps = 1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = tf.placeholder(tf.float32, shape=[None,input_dim], name = "inputs")
        self.actions = tf.placeholder(tf.float32, shape =[None,], name = "actions")
        self.advantages = tf.placeholder(tf.float32, shape=[None,], name = "advantages")
        self.hidden_layers = []
        self.training_rate = training_rate
        self.min_experiences = 100
        self.max_experiences = 1000
        self.batch_size = 32
        self.inner_counter = 0.0
        self.eps = eps
        self.experiences = {'states': [], 'actions': [], 'advantages': []}
        idim = input_dim
        if use_res_net: # Then fill out tensorboard with res net
            for i in range(len(hl_sizes)):
                if i == 0:
                    self.hidden_layers.append(FullyConnectedLayer(idim, hl_sizes[i], activation_fun = activation_fun))
                else:
                    self.hidden_layers.append(FullyConnectedResNetBlock(idim, [hl_sizes[i]], activation_fun = activation_fun, batch_normalization = False))
                idim = hl_sizes[i]

        else: # Use regular fully connected layers
            for hl in hl_sizes:
                self.hidden_layers.append(FullyConnectedLayer(idim, hl, activation_fun = activation_fun))
                idim = hl
        # last layer is Regressive to single node -- One for Mean, One for Std_dev
        self.Y_mean = FullyConnectedLayer(idim, output_dim, activation_fun = None)
        self.Y_std = FullyConnectedLayer(idim, output_dim, activation_fun= tf.nn.relu)
        
        # Rollout Abstraction
        Z = self.X
        for hl in self.hidden_layers:
            Z = hl.forward(Z)
        #Mean
        mean = tf.reshape(self.Y_mean.forward(Z), [-1]) 
        std = tf.reshape(self.Y_std.forward(Z), [-1]) + 1e-5

        # Sample from the normal distribution
        norm = tf.contrib.distributions.Normal(mean, std)
        self.predict_op = tf.clip_by_value(norm.sample(), -1,1)

        log_probs = norm.log_prob(self.actions)
        self.cost = -tf.reduce_sum(self.advantages * log_probs + .1*norm.entropy())
        self.train_op = tf.train.AdamOptimizer(self.training_rate).minimize(self.cost)
    def set_session(self, sess):
        self.session = sess

    def partial_fit(self, X, actions, advantages, printOp = True):
        #X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        #print(X, actions, advantages)
        self.experiences['actions'].append(actions)
        self.experiences['advantages'].append(advantages)
        self.experiences['states'].append(X)
        if len(self.experiences['advantages']) < self.min_experiences:
            return
        if len(self.experiences['advantages']) > self.max_experiences:
            self.experiences['advantages'].pop(0)
            self.experiences['actions'].pop(0)
            self.experiences['states'].pop(0)
        indxs = np.random.choice(len(self.experiences['advantages']), self.batch_size)
        #print("INDEXES: ", indxs)
        states = [self.experiences['states'][indx] for indx in indxs]
        states = np.reshape(np.array(states), (self.batch_size, self.input_dim))
        actions = [self.experiences['actions'][indx] for indx in indxs]
        actions = np.reshape(actions, (self.batch_size,))
        advantages = [self.experiences['advantages'][indx] for indx in indxs]
        advantages = np.reshape(advantages, (self.batch_size,))
        loss, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: states, self.actions: actions, self.advantages: advantages})
        if printOp:
            print("Computed Partial Fit with Loss of: ", loss)
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})
    
    def sample_action(self, X):
        return np.asscalar(self.predict(X)[0])


# For a Fully Connected Nueral Network
class ValueModel:
    def __init__(self, input_dim, output_dim, hidden_layer_sizes = [64], activation_fun = tf.nn.relu, use_res_net_blocks = True, training_rate = 1e-3):
        self.X = tf.placeholder(tf.float32, shape=[None,input_dim], name = "inputs")
        self.Y = tf.placeholder(tf.float32, shape =[None,], name = "actions")
        self.hidden_layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.training_rate = training_rate
        idim = input_dim
        self.hl_sizes = hidden_layer_sizes
        self.experiences = {'states':[], 'values': []}
        self.min_experiences = 100
        self.max_expereinces = 1000
        self.batch_size = 32
        self.Xs = []
        self.Ys = []
        if use_res_net_blocks: # Then fill out tensorboard with res net
            for i in range(len(hl_sizes)):
                if i == 0:
                    self.hidden_layers.append(FullyConnectedLayer(idim, hl_sizes[i], activation_fun = activation_fun))
                else:
                    self.hidden_layers.append(FullyConnectedResNetBlock(idim, [hl_sizes[i]], activation_fun = activation_fun, batch_normalization = False))
                idim = hl_sizes[i]

        else: # Use regular fully connected layers
            for hl in self.hl_sizes:
                self.hidden_layers.append(FullyConnectedLayer(idim, hl, activation_fun = activation_fun))
                idim = hl
        # Computes the Value of the current state 
        self.h_last = FullyConnectedLayer(idim, output_dim, activation_fun = None)

        # Graph abstraction
        Z = self.X
        for hl in self.hidden_layers:
            Z = hl.forward(Z)
        self.Y_pred = self.h_last.forward(Z)
        self.Y_pred = tf.reshape(self.Y_pred, [-1])

        # Cost
        self.cost = tf.reduce_sum(tf.square(self.Y - self.Y_pred))
        self.train_op = tf.train.AdamOptimizer(self.training_rate).minimize(self.cost)
    def set_session(self, sess):
        self.session = sess

    def partial_fit(self, X, Y, printOp = True):
        self.experiences['states'].append(X)
        self.experiences['values'].append(Y)
        if len(self.experiences['states']) < self.min_experiences:
            return
        if len(self.experiences['states']) > self.max_expereinces:
            self.experiences['states'].pop(0)
            self.experiences['values'].pop(0)
        indxs = np.random.choice(len(self.experiences['states']), self.batch_size)
        #print("INDEXES: ", indxs)
        X = [self.experiences['states'][indx] for indx in indxs]
        X = np.reshape(np.array(X), (self.batch_size, self.input_dim))
        Y = [self.experiences['values'][indx] for indx in indxs]
        Y = np.reshape(np.array(Y), (self.batch_size,))
        loss, _ = self.session.run([self.cost, self.train_op], feed_dict = {self.X: X, self.Y: Y})
        if printOp:
            print("Partial Fit Loss is: ", loss)
    
    def predict(self, X):
        return self.session.run(self.Y_pred, feed_dict = {self.X: X})


actions = []
rounds = 500
max_steps = 250
done = False
idim = 2
odim = 1
hl_sizes = [32]
use_res_net = True
sess = tf.Session()
Policy = PolicyModel(input_dim = idim, output_dim=odim, hl_sizes=hl_sizes, use_res_net=use_res_net)
Value = ValueModel(input_dim=idim, output_dim=odim, hidden_layer_sizes=[32], use_res_net_blocks=True)
sess.run(tf.global_variables_initializer())
Policy.set_session(sess)
Value.set_session(sess)
gamma = .95
count = 0
total_rewards = []
for r in range(rounds):
    done = False
    count = 0
    total_reward = 0
    state = env.reset()
    print("Round ", r)
    while not done and count < max_steps:
        #env.render()
        a = Policy.sample_action(state)
        #print("Action Value: ", a)
        new_state, reward, done, _ =  env.step([a])
        #print(new_state, reward, done)
        new_state = np.reshape(new_state, (1,2))
        state = np.reshape(new_state, (1,2))
        total_reward += reward
        value_new_state = Value.predict(new_state)
        G = reward + gamma*value_new_state
        advantage = G - Value.predict(state)
        Policy.partial_fit(state, a, advantage, printOp = False)
        Value.partial_fit(state, G, printOp = False)
        state = new_state
        count += 1
    print("Total Round Reward: ", total_reward)
    total_rewards.append(total_reward)
plt.figure(1)
plt.title("Total rewards over episodes")
plt.plot([i for i in range(rounds)],total_rewards)
env.close()
