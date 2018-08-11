# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:30:01 2018

@author: natsn
"""

import numpy as np
import tensorflow as tf
import time 

# a version of HiddenLayer that keeps track of params
class HiddenLayer:
    
    def __init__(self, size_of_input, size_of_layer, f = tf.nn.tanh):
        
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
        a = tf.add(tf.matmul(X, self.W),self.b)
        return self.f(a) 


class DQN:
    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes, gamma,
                 max_experiences=10000, min_experiences=200, batch_sz=96, learning_rate = 1e-3):

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
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        
        #self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(self.cost)
        #self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(self.cost)
        
        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.total_rewards = []
        self.total_losses = []
        

    # Instantiate the session from outside of the model
    def set_session(self, session):
        self.session = session

    # Copy the model to the copy model, which is used to run predictions
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
        print('Graph Size Before Update: ')
        get_tf_variable_size()
        self.session.run(ops)
        print('New Graph Size: ')
        get_tf_variable_size()

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
            # don't do anything if we don't have enough experience
            return None

        # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False) # returns a list of positional indexes
        
        states = [self.experience['s'][i] for i in idx]
        states = self.flatten_states(states)
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        next_states = self.flatten_states(next_states)
        dones = [self.experience['done'][i] for i in idx]
        
        # With our SARS' fourple, based on our initial state, we will take the next state the action landed us in (s') and compute the maximum next state reward we can achieve
        # It is very important that we call the predict function on the target_network
        max_ns_rewards = np.max(target_network.predict(next_states), axis=1)
        #max_ns_reward = target_network.predict(next_states) # Currently we only have 1 output guess
        
        # Targets, aka the current hypothesized return from a state, is what we iteratively train on.
        targets = [r + self.gamma*mnsr if not done else r for r, mnsr, done in zip(rewards, max_ns_rewards, dones)]

        # Call the optimizer. Predict the loss from the current batch using the return as the target goal, with respect to THAT action the agent has chosen
        #self.session.run(self.train_op,feed_dict= {self.X: states, self.G: targets})
        loss, _ = self.session.run([self.cost, self.train_op], feed_dict= {self.X: states, self.G: targets, self.actions: actions})
        return loss
        
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
    def pget_action_argnum(self, x, eps):
        if np.random.random() < eps:
            print('EPSILON-GREEDY RANDOM ACTION SELECTED')
            return np.random.choice(self.n_outputs) # Choose a random action 0,1
        else:
            X = np.atleast_2d(x)
            tic = time.time()
            act_arg = np.argmax(self.predict(X))
            toc = time.time()
            print("TIC TOC TIME:  ", toc - tic)
            return act_arg # returns the argument number of the highest return found from the Q-Net
            #return self.predict(x) # returns the argyment number of highest return
    
    # Sample action from our agents NN
    def sample_action(self,obs4,eps):
        # reshape the stacked state information to a flattened state
        obs4flat = obs4.reshape((1,np.size(obs4)))
        actionNum = self.pget_action_argnum(obs4flat,eps)
        action = None
        
        if actionNum == 0:
            action = 'No_Intervention'
        elif actionNum == 1:
            action = 'Intervention'
        
        return action, actionNum    # Action represents the offset we will give to the quadcopter
    
    def state_stack(self, obs4, obs): 
        # will only occur if obs4 is empty..ie after a reset.
        if len(obs4) == 0:
            obs = np.atleast_2d(obs)
            obs4 = obs
            obs4 = np.append(obs4,obs, axis = 0)
            obs4 = np.append(obs4,obs, axis = 0)
            obs4 = np.append(obs4,obs, axis = 0)
            return obs4
        else:
            # pop top item, append newest item to back
            obs4 = np.append(obs4[1:],np.atleast_2d(obs),axis = 0)
            return obs4
        
    def flatten_states(self,states):
        flat_len = np.size(states[0])
        states = [np.atleast_2d(s.reshape(1,np.size(s))) for s in states]
        states = np.array(states).reshape((self.batch_sz,flat_len))
        return states



def get_tf_variable_size():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)







class PIDCutoff:
    def __init__(self, max_roll, max_pitch, minimax_pos_vel_z = (2,-2)):

        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.total_rewards = []
        self.max_roll = max_roll
        self.max_pitch = max_pitch
        self.minimax_pos_vel_z = minimax_pos_vel_z
        
    
    # Sample action from our agents NN
    def sample_action(self,obs):
        action = None
        actionNum = None
        #Look at roll threshold
        if np.abs(obs[0]) > self.max_roll:
        #Look at pitch threshold
            action = 'Intervention'
            actionNum = 1
        elif np.abs(obs[1]) > self.max_pitch:
            action = 'Intervention'
            actionNum = 1
        #Look at posZ
        elif obs[2] < self.minimax_pos_vel_z[0] and obs[3] < self.minimax_pos_vel_z[1]:
            print("TEST!!!",obs[2], obs[3])
            action = 'Intervention'
            actionNum = 1
        else:
            action = 'No_Intervention'
            actionNum = 0
        
        return action, actionNum    # Action represents the offset we will give to the quadcopter
    







