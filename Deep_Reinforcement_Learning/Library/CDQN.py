# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:00:44 2018

@author: natsn
"""


import tensorflow as tf
import numpy as np
import sys
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Neural_Network")
sys.path.append("D:\Desktop\Research\Machine_Learning\Anaconda\Spyder\Reinforcement_Learning_Master\Convolutional Neural Networks (CNN)")
import FullyConnectedLayer as FCL
import CNNBlocks
import time as time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class CDQN():
    def __init__(self, x_dim, n_outputs, hidden_layer_sizes, gamma,
                 max_experiences = 10000, min_experiences = 100, 
                 batch_sz= 32, learning_rate = 1e-3):

        #Parameters for tuning
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma
        
        #Input and Output Layer dimentions
        self.x_dim = x_dim
        self.n_outputs = n_outputs
        self.train_dim = (self.batch_sz, x_dim[0], x_dim[1], x_dim[2])
        
        # Declare Tensor Placeholders
        Image_Array_Dim = [None] + list(x_dim)
        self.X = tf.placeholder(tf.float32, shape = Image_Array_Dim, name='X') # None = it can vary
        self.G = tf.placeholder(tf.float32, shape=(batch_sz,), name='G') # a return of the given state
        self.actions = tf.placeholder(tf.int32, shape=(batch_sz,), name='actions') # a list of actions per each input state
        
        # Create Graph Abstraction for Convolutional Layers
        self.CNN_Block_Layers = []
        self.pool_stride_block_settings = {'Num_Pools': [], 'Conv_Stride': [] ,'Pool_Stride': []}
        self.FC_layers = []
        
        self.FC_params = [] # Storage of FC Param Variables for tensor copies
        self.CNN_params = [] # Storage of FC Param Variables for tensor copies
        
        # Stack Convolutional Blocks -- Most of VGG Model
        ConvBlock = CNNBlocks.VGGConvPoolBlock32()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock32()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock64()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock64()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        # Determine Number of Stacked Convolutional Blocks
        self.num_conv_blocks = len(self.CNN_Block_Layers)
        
        # Initialize Proper Convolutional Kernal Dimentionality
        idim = self.x_dim[2]
        print("Input Dim: ", idim)
        for i in range(self.num_conv_blocks):
            # Change the input of a block layer to have the same input channel dimention as the inputted image / feature map channel
            self.CNN_Block_Layers[i].set_layer_input_channel_dim(idim) # Layer zero or 1
            idim = self.CNN_Block_Layers[i].get_layer_output_channel_dim(self.CNN_Block_Layers[i].layer_count - 1) # Layer Zero or Layer 1 ( we choose layer one to connect to the next block)
            
            # Store Pool Stride information
            self.pool_stride_block_settings['Num_Pools'].append(self.CNN_Block_Layers[i].get_num_pools())
            self.pool_stride_block_settings['Pool_Stride'].append(self.CNN_Block_Layers[i].get_block_pool_stride())
           
        
        # Get input size for the fully connected layer
        FC_INPUT_SIZE = int(self.get_FC_input_size(x_dim,
                                               self.pool_stride_block_settings['Num_Pools'],
                                               self.pool_stride_block_settings['Pool_Stride']))
        
        # Begin Graph Abstraction for Fully Connected Layer
        input_size = FC_INPUT_SIZE
        
        # Declare Fully Connected Layers
        for layer_size in hidden_layer_sizes:
            FC_layer = FCL.FullyConnectedLayer(input_size, layer_size)
            self.FC_layers.append(FC_layer)
            input_size = layer_size
        # Append Final Fully Connected layer
        FC_layer = FCL.FullyConnectedLayer(input_size, self.n_outputs, activation_fun = None) # final layer does not take an activation function
        self.FC_layers.append(FC_layer)
        
        # Collect fully connected params for copy model's FC Weights
        for layer in self.FC_layers:
            self.FC_params += layer.params # concatenate all members into params
        
        # Collect convolutional params for copy model's FC Weights
        for Block in self.CNN_Block_Layers:
            for layer in Block.conv_layers:
                self.CNN_params += layer.params # concatenate all members into params
        
        # Rollout Graph Calculations:
        # (1/2) Rollout Convolutional Calculations:
        Z = self.X
        for i in range(self.num_conv_blocks):
            Z = self.CNN_Block_Layers[i].forward(Z)
        
        # Reshape Z for the Fully Connected Rollout
        Z = tf.reshape(Z,(-1,FC_INPUT_SIZE))
        
        # Fully Conneccted Rollout
        for i in range(len(hidden_layer_sizes)):
            Z = self.FC_layers[i].forward(Z)
        # Last layer -- Non activated
        self.predict_op = self.FC_layers[-1].forward(Z)        
        
        # Network's action selection. Predicts the VALUE of the action selection we chose:
        self.selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, self.n_outputs),reduction_indices=[1])
        
        # Take the targets calculated by the Target network and subtract from our network's prediction
        self.cost = tf.reduce_sum(tf.square(self.G - self.selected_action_values))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        #self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(self.cost)
        #self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(self.cost)
        
        # Create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.total_rewards = []
        self.total_losses = []
        
        print("Tensorflow Graph Abstraction Instantiated")
        
    # Instantiate the session from outside of the model
    def set_session(self, session):
        self.session = session
    def restore_session(self, filedir):
        saver = tf.train.Saver()
        saver.restore(self.session, filedir + "\\model.ckpt")
        print("Session Restored!")
    def save_session(self, filedir):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, filedir + "//model.ckpt")
        print("Model saved in path:", save_path)
        
    # Copy the model to the copy model, which is used to run predictions on
    def copy_from(self, other):
        
        # collect all ops
        ops = []
        my_params = self.FC_params
        other_params = other.FC_params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        
        my_params = self.CNN_params
        other_params = other.CNN_params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        
        # Now run them all
        print('Graph Size Before Update: ')
        get_tf_variable_size()
        self.session.run(ops)
        print('New Graph Size: ')
        get_tf_variable_size()

    def predict(self, x):
        print("X_Predict Shape: ", x.shape)
        #X = np.float32(x)
        tic = time.time()
        pred = self.session.run(self.predict_op, feed_dict={self.X: x})
        print("Prediction Time: ", time.time() - tic)
        return pred
        
    def train(self, target_network, iterations = 1):
        # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
            # don't do anything if we don't have enough experience
            return None
        loss = 0
        

        for i in range(iterations):
            tic = time.time()
            # randomly select a batch
            idx = np.random.choice(len(self.experience['s']), size = self.batch_sz, replace=False) # returns a list of positional indexes
            
            states = np.array([self.experience['s'][i] for i in idx], np.float32)
            next_states = np.array([self.experience['s2'][i] for i in idx], np.float32)
            
            states.reshape(self.batch_sz, self.x_dim[0], self.x_dim[1], self.x_dim[2])
            next_states.reshape(self.batch_sz, self.x_dim[0], self.x_dim[1], self.x_dim[2]) #Deoesn't have self.x_dim[2]
        
            actions = [self.experience['a'][i] for i in idx]
            rewards = [self.experience['r'][i] for i in idx]
            dones = [self.experience['done'][i] for i in idx]
            
            # With our SARS' fourple, based on our initial state, we will take the next state the action landed us in (s') and compute the maximum next state reward we can achieve
            # It is very important that we call the predict function on the target_network
            max_ns_rewards = np.max(target_network.predict(next_states), axis = 1)
            #max_ns_reward = target_network.predict(next_states) # Currently we only have 1 output guess
            
            # Targets, aka the current hypothesized return from a state, is what we iteratively train on.
            targets = [r + self.gamma*mnsr if not done else r for r, mnsr, done in zip(rewards, max_ns_rewards, dones)]
    
            # Call the optimizer. Predict the loss from the current batch using the return as the target goal, with respect to THAT action the agent has chosen
            #self.session.run(self.train_op,feed_dict= {self.X: states, self.G: targets})
            loss, _ = self.session.run([self.cost, self.train_op], feed_dict= {self.X: states, self.G: targets, self.actions: actions})
            
            print("Stochastic Train Round: ", i, "Loss: ", loss, ", Time: ", time.time() - tic)
        
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
            tic = time.time()
            X = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
            act_arg = np.argmax(self.predict(X))
            toc = time.time()
            print("Prediction: ", act_arg, "Time: ", toc - tic)
            return act_arg # returns the argument number of the highest return found from the Q-Net
            #return self.predict(x) # returns the argyment number of highest return
    
    # Sample action from our agents NN
    def sample_action(self,obs4,eps):
        action = self.pget_action_argnum(obs4,eps)
        return action
    
    # We need to assume that our convolutions are 
    def get_FC_input_size(self, img_height_width, num_pools, Pool_Strides):
        
        # Initialize
        img_sizes = list(img_height_width)
        
        # This would have to change to deal with variable size and shaped tensors, but good for 2D Variable stride tensors
        for i in range(self.num_conv_blocks):
            # Reduce dimentionaility per the pool instructions
            if num_pools[i] != 0:
                img_sizes[0] /= (Pool_Strides[i][1] * num_pools[i])
                img_sizes[1] /= (Pool_Strides[i][2] * num_pools[i])
        
        return img_sizes[0] * img_sizes[1] * self.CNN_Block_Layers[-1].get_block_out_dim()
        

class CRDQN():
    def __init__():
        pass # Add
    



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
















