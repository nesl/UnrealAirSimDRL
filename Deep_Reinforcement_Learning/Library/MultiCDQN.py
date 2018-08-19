# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 21:28:16 2018

@author: natsn
"""

import CDQN

# Used to control a swarm, basically can set all CQDNs here at once
class MultiCDQN:
    
    def __init__(self, num_networks, xdim, n_outputs, 
                 hidden_layer_sizes, gamma, 
                 max_experiences, min_experiences,
                 batch_size, learning_rate  ):
        self.num_networks = num_networks
        self.ConvDQNs = [CDQN.CDQN(xdim, n_outputs, hidden_layer_sizes, 
                                   gamma, max_experiences, min_experiences, 
                                   batch_size, learning_rate) for i in range(num_networks)]
    # Instantiate the session from outside of the model
    def set_sessions(self, session):
        for i in range(self.num_networks):
            self.ConvDQNs[i].set_session(session)
            
    def restore_session(self, filedirs):
        assert len(filedirs) == self.num_networks
        for i, fd in zip(filedirs, range(self.num_networks)):
            self.ConvDQNs[i].restore_session(fd)
            
    def save_sessions(self, filedirs):
        assert len(filedirs) == self.num_networks
        for i, fd in zip(filedirs, range(self.num_networks)):
            self.ConvDQNs[i].save_session(fd)
        
    # Copy the model to the copy model, which is used to run predictions on
    def copy_froms(self, others):
        assert len(others) == self.num_networks
        for i, oth in zip(others, range(self.num_networks)):
            self.ConvDQNs[i].copy_from(oth)
            
    def predicts(self, Xs):
        assert len(Xs) == self.num_networks
        data = []
        for i, x in zip(Xs, range(self.num_networks)):
            data.append(self.ConvDQNs[i].predict(x))
        return data
        
    def trains(self, target_networks, iterations = 1):
        
        assert len(target_networks) == self.num_networks
        data = []
        for i, x in zip(target_networks, range(self.num_networks)):
            data.append(self.ConvDQNs[i].train(x, iterations))
        return data
        
    # This function will run each and every time we take a step inside our model.
    def add_experiences(self, Ss, As, Rs, S2s, dones):
        assert len(Ss) == self.num_networks
        for s,a,r,s2,done, i in zip(Ss, As, Rs, S2s, dones, range(self.num_networks)):
            self.ConvDQNs[i].add_experience(s,a,r,s2,done)
    
    # Use decaying epsilon greedy to converge onto our policy
    def pget_action_argnums(self, Xs, ep):
        assert len(Xs) == self.num_networks
        data = []
        for i, x in zip(Xs, range(self.num_networks)):
            data.append(self.ConvDQNs[i].pget_action_argnum(x, ep))
        return data
        
    # Sample action from our agents NN
    def sample_actions(self,obs4s,eps):
        actions = self.pget_action_argnums(obs4s,eps)
        return actions
    




# Used to control a swarm, basically can have 1 CDQN
# Everything is received and transmitted in dictionaries, per each vehicle used in simulation
class MasterCDQN:
    
    def __init__(self, vehicle_names, xdim, n_outputs, 
                 hidden_layer_sizes, gamma, 
                 max_experiences, min_experiences,
                 batch_size, learning_rate):
        self.vehicle_names = vehicle_names
        self.CDQN = CDQN.CDQN(xdim, n_outputs, hidden_layer_sizes, 
                                   gamma, max_experiences, min_experiences, 
                                   batch_size, learning_rate)
    # Instantiate the session from outside of the model
    def set_session(self, session):
        self.CDQN.set_session(session)
            
    def restore_session(self, filedir):
        self.CDQN.restore_session(filedir)
            
    def save_session(self, filedir):
        self.CDQN.save_session(filedir)
        
    # Copy the model to the copy model, which is used to run predictions on
    def copy_from(self, other):
        self.CDQN.copy_from(other)
    
    # Pass in a list of inputs
    # Inputs should be a dictionary with each x in the corresponding vehicle bucket
    def predicts(self, Xs):
        assert len(Xs) == len(self.vehicle_names)
        data = dict.fromkeys(self.vehicle_names)
        
        for vn in self.vehicle_names:
            data[vn] = self.CDQN.predict(Xs[vn])
        return data
        
    def train(self, target_network, iterations = 1):
        return self.CDQN.train(target_network, iterations)
        
    # This function will run each and every time we take a step inside our model.
    def add_experiences(self, Ss, As, Rs, S2s, dones):
        assert len(Ss) == len(self.vehicle_names)
        for vn in self.vehicle_names:
            self.CDQN.add_experience(Ss[vn], As[vn], Rs[vn], S2s[vn], dones[vn])
    
    # Use decaying epsilon greedy to converge onto our policy
    def pget_action_argnums(self, Xs, ep):
        assert len(Xs) == len(self.vehicle_names)
        data = dict.fromkeys(self.vehicle_names)
        for vn in self.vehicle_names:
            data[vn] = self.CDQN.pget_action_argnum(Xs[vn], ep)
        return data
        
    # Sample action from our agents NN
    def sample_actions(self,obs4s,eps):
        actions = self.pget_action_argnums(obs4s,eps)
        return actions


















