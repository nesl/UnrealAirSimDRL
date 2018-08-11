# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:16:38 2017
@author: natsn
"""
# Notes: This script will iteratively learn to self balance, updated to have max x-y drift
# Be careful about the z direction as negative really means fly upwards

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from q_learning_bins import plot_running_avg

import AirSimClient as ASC

class environment:
    # setpoint is where we would like the drone to stabilize around
    # minimax_z_thresh should be positive and describe max and min allowable z-coordinates the drone can be in
    # max drift is how far in x and y the drone can drift from the setpoint in x and y without episode ending (done flag will be thrown)
    def __init__(self, setpoint, minimax_z_thresh, max_xy_drift = (2,2)):
        
        # States will be pitch, roll, vx,vy,vz
        # Number of states = 5
        s = np.zeros(8)
        self.current_state = np.array(s)
        
        # Initialize starting position
        initX =  0 # -.55265
        initY =  0 # -31.9786
        initZ =  -1.5 # -19.0225
        
        
        self.initial_position  = (initX,initY,initZ)
        assert setpoint[2] <= 0
        self.setpoint = setpoint
        assert minimax_z_thresh[0] >= 0 and minimax_z_thresh[1] >= 0 and minimax_z_thresh[0] < minimax_z_thresh[1]
        self.minimax_z_thresh = minimax_z_thresh
        assert max_xy_drift[0] > 0 and max_xy_drift[1] > 0
        self.max_xy_drift = max_xy_drift
        
        
        # connect to the AirSim simulator 
        print('Initializing Client')
        self.client = ASC.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print('Initialization Complete!')
        #self.client.moveByVelocity(1, -0.67, -0.8, 5)
        
    # The action will be to Go Left / Right / Up / Down for 50 ms
    def step(self, action):
        # If the try block fails, then this means we have crashed the drone
        
        # 1. Take action in the simulator based on the DRL predict function
        self.do_action(action)
        
        # 2. Get current state if not done
        self.current_state = self.pget_simulator_state_info()
        
        # 3. Collect reward, check done flag from taking action
        # reward function uses x y and z positions 
        reward, done = self.calc_reward(self.current_state[0:3])
        
        return (self.current_state,reward,done)
            
    
    # Ask the simulator to return attitude information (private)
    def pget_simulator_state_info(self):
        pos = self.client.getPosition()
        vel = self.client.getVelocity()
        pry = self.client.getPitchRollYaw()
        
        #unpack and repack all state information into our eight state input
        state = np.array([pos.x_val,pos.y_val,pos.z_val,vel.x_val,vel.y_val,vel.z_val,pry[0],pry[1]])
        return state
    
    #Ask the simulator to perform control command or block control command
    def do_action(self, quad_offset): 
        quad_vel = self.client.getVelocity()
        duration = .15
        self.client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], duration)
        time.sleep(0.1)
        
    # The reward at every timestep will be around zero
    def calc_reward(self, stateXYZ):
        stateX = stateXYZ[0]
        stateY = stateXYZ[1]
        stateZ = -1*stateXYZ[2] # make positive so math is easier
        
        # if we are above or below our z threshold, throw done flag
        if stateZ > self.minimax_z_thresh[1] or stateZ < self.minimax_z_thresh[0]:
            done = True
            reward = -200
        # If we have drifted sideways too much in x or y, throw done flag
        elif np.abs(stateX) > np.abs(self.initial_position[0] - self.max_xy_drift[0]) or np.abs(stateY) > np.abs(self.initial_position[1] -self.max_xy_drift[1]):
            done = True
            reward = -200
        # If we are still in our limits, then compute reward function and continue
        else: # reward based on spatial location of quadcopter to the setpoint (X,Y,Z) -> Z is negated so it is positive
            done = False
            reward = -2*(self.setpoint[0] - stateX)**2 + -2*(self.setpoint[0] - stateY)**2 + -2*(np.abs(self.setpoint[2]) - stateZ)**2
        return reward, done
        
    def reset(self):
        #self.client.reset()
        #time.sleep(.25)
        print('Reseting Quad')
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        #self.client.armDisarm(True)
        
        # Option 1
        #self.client.takeoff()
        
        # Option 2 -- THINK THIS WILL WORK FASTER
        # Quickly raise the quadcopter to 1m -- Speed up
        for i in range(11):
            quad_vel = self.client.getVelocity()
            duration = .15
            quad_offset = (0, 0, -.15)
            self.client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], duration)
            time.sleep(.1)
            xyz = self.client.getPosition()
        
        # Quickly raise the quadcopter to 1m -- Slow down
        for i in range(7):
            quad_vel = self.client.getVelocity()
            duration = .15
            quad_offset = (0, 0, .15)
            self.client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], duration)
            time.sleep(.1)
            xyz = self.client.getPosition()
        
        quad_vel = self.client.getVelocity()
        
        print('Z Init Vel: ', quad_vel.z_val)
        print('Z Init Position: ', xyz.z_val)
        # End of Option 2
        
        
        #self.client.moveToPosition(self.initial_position[0], self.initial_position[1], self.initial_position[2], 1)
        print('Reset Complete')
        
        # self.client.moveByVelocity(1, -0.67, -0.8, 5)
        pos = self.client.getPosition()
        vel = self.client.getVelocity()
        pry = self.client.getPitchRollYaw()
        
        # unpack and repack all state information into our eight state input
        init_state = np.array([pos.x_val,pos.y_val,pos.z_val,vel.x_val,vel.y_val,vel.z_val,pry[0],pry[1]])
        
        return init_state

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
        #self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        #self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
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
    def pget_action_argnum(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.n_outputs) # Choose a random action 0,1,2,3
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)) # returns the argument number of the highest return found from the Q-Net
            #return self.predict(x) # returns the argyment number of highest return
    
    # Sample action from our agents NN
    def sample_action(self,x,eps):
        scaling_factor = 0.15
        # input state and epsilon (for search) 
        actionNum = self.pget_action_argnum(x,eps)
        if actionNum == 0:
            action = (0, 0, 0)
        elif actionNum == 1:
            action = (scaling_factor, 0, 0)
        elif actionNum == 2:
            action = (0, scaling_factor, 0)
        elif actionNum == 3:
            action = (0, 0, scaling_factor)
        elif actionNum == 4:
            action = (-scaling_factor, 0, 0)    
        elif actionNum == 5:
            action = (0, -scaling_factor, 0)
        elif actionNum == 6:
            action = (0, 0, -scaling_factor)
        return action, actionNum    # Action represents the offset we will give to the quadcopter
        
        
    # plays one round of the environment you pass in
    def play_game(self, TrainingModel, env, rounds = 1, copy_period = 40):
        print('Begin Learning!')
        for i in range(rounds):
            # Decaying Epsilon Greedy
            ep = self.epsilon / np.sqrt(i+1)
            if ep < .05:
                ep = .05
            
            # Print out some Statistics
            if i % 25 == 0 and i != 0:
                print("Round:", i, "total reward:", np.sum(self.total_rewards), "eps:", ep, "avg reward (last 25):", np.average(self.total_rewards[-25:]))
                saver = tf.train.Saver()
                saver.save(self.session,"./saved_NN_models/DQN_Stability_V2.ckpt")
            else:
                print("Round ", i)
            
            # Reset Variables for next round
            obs = env.reset() 
            done = False
            totalreward = 0
            local_round = 0
            it = 0
            
            while not done and it < 350:
                local_round += 1
                action, actionNum = self.sample_action(obs, ep)
                prev_obs = obs
                obs, reward, done = env.step(action)
                
                totalreward += reward
            
                # Add memory to memory replay dictionary and train model using the training model
                print('local round:', local_round,'state: Z: ', obs[2],'  ', 'action: ', actionNum, 'reward: ', reward )
                self.add_experience(prev_obs, actionNum, reward, obs, done)
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
    setpoint = (0,0,-2)
    max_xy_drift = (1.5,1.5)
    minimax_z_thresh = (.01, 3.5) # in absolute z-coords
    
    env = environment(setpoint,minimax_z_thresh,max_xy_drift)

    n_inputs = 8 # roll, pitch, Vx, Vy, Vz, Pos x, y, z 
    n_outputs = 7 # L/R/F/B/U/D/nothing
    sizes = [200,200]
    gamma = .98

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
    N = 1000
    totalrewards = model.play_game(tmodel, env, rounds = N)

    # Observe how well the model converged to optimal
    #for i in range(10):
        #watch_episode(model, env)

    plt.scatter([i for i in range(N)],totalrewards)
    plt.title("Rewards")
    plt.show()


if __name__ == '__main__':
    main()
  
  
  
  
  
  
  
  
  
  
  
  