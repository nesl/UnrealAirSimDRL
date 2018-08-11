# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 00:31:36 2018

@author: natsn
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    
    def __init__(self, dim_inp, dim_hl):
        
        self.X = tf.placeholder(dtype = tf.float32, shape = [None,dim_inp])
        
        W1 = tf.Variable(tf.random_normal(shape = [dim_inp,dim_hl]), dytpe = tf.float32)
        b1 = tf.Variable(tf.random_normal(dim_hl,1), dytpe = tf.float32)
        
        W2 = tf.Variable(tf.random_normal(dtype = tf.float32, shape = [dim_hl,dim_inp]), dytpe = tf.float32) 
        b2 = tf.Variable(tf.random_normal(dim_inp,1), dytpe = tf.float32)
        
        # COnstruct graph
        Z = tf.nn.relu(tf.add(tf.matmul(self.X,W1),b1))
        self.logits = tf.add(tf.matmul(Z,W2),b2)
        self.X_hat = tf.nn.sigmoid(self.logits)
        
        self.cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.X))
        self.optimize = tf.train.AdamOptimizer(.001).minimize(self.cost)
    
    # Input X as a N sample by M dimention image vector
    def train(self, X):
        # Take minibatches of the X and train on epochs of it
        batch_sz = 200 # change
        epochs = 50
        for e in range(ep):  
            rand_indx = np.random.choice(len(X), size = batch_sz)
            X_mb = X[rand_indx,:]
            self.sess.run([self.cost, self.optimize], feed_dic = {self.X : X})
            
        
    def set_session(self, session):
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

        
        
        
        