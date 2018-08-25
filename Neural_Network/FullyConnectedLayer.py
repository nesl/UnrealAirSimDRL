# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 01:47:20 2018

@author: natsn
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\Util")
import tensorflow as tf


# Fully Connected hidden layer
class FullyConnectedLayer():

    def __init__(self, inp_sz, op_sz, bias = True, activation_fun = tf.tanh):
        
        # Tensor Sizes, Bias, and activation function
        self.inp_sz = inp_sz
        self.op_sz = op_sz
        self.bias = bias
        self.activation_fun = activation_fun

        # For Fully Connected Tensor Ops
        self.W = tf.Variable(tf.random_normal(shape = [inp_sz,op_sz]))
        self.b = tf.Variable(tf.random_normal(shape = [1,op_sz]))
        
        # Store Params in list
        self.params = [self.W] + [self.b]


    def forward(self, Z):
        if self.bias:
            if self.activation_fun == None:
                return tf.add(tf.matmul(Z,self.W), self.b)
            else:
                return self.activation_fun(tf.add(tf.matmul(Z,self.W), self.b))
        else: 
            if self.activation_fun == None:
                return tf.matmul(Z,self.W)
            else:
                return self.activation_fun(tf.matmul(Z,self.W))
   