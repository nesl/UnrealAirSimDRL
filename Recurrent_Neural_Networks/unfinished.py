# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:22:00 2018

@author: natsn
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Simple ANN Script with TF
input_size = 10
output_size = 10
hl_sizes = [400,400,400]

# Instantiate input and output
X = tf.placeholder(dtype = tf.float32, shape = (None,input_size))
Yt = tf.placeholder(dtype = tf.float32, shape = (None,output_size))


# Initialize Weights and Biases
W1 = tf.Variable(tf.random_normal(shape = (input_size,hl_sizes[0])))
b1 = tf.Variable(tf.random_normal(shape = (1,hl_sizes[0])))
W2 = tf.Variable(tf.random_normal(shape = (hl_sizes[0],hl_sizes[1])))
b2 = tf.Variable(tf.random_normal(shape = (1,hl_sizes[1])))
W3 = tf.Variable(tf.random_normal(shape = (hl_sizes[1],hl_sizes[2])))
b3 = tf.Variable(tf.random_normal(shape = (1,hl_sizes[2])))
W4 = tf.Variable(tf.random_normal(shape = (hl_sizes[2],output_size)))
b4 = tf.Variable(tf.random_normal(shape = (1,output_size)))

# Declare Network Abstraction
Z1 = tf.add(tf.matmul(X,W1),b1)
Z1a = tf.tanh(Z1)
Z2 = tf.add(tf.matmul(Z1a,W2),b2)
Z2a = tf.tanh(Z2)
Z3 = tf.add(tf.matmul(Z2a,W3),b3)
Z3a = tf.tanh(Z3)
Z4 = tf.add(tf.matmul(Z3a,W4),b4)

# If we have a classifier we can use the softmax layer at the end
y_op = tf.nn.softmax(labels = Yt, logits = Z4, dim = 1)

# Cost 
cost = tf.reduce_sum(y_op = tf.nn.softmax_cross_entropy_with_logits(labels = Yt, logits = Z4, dim = 1))

# Optimize
optimizer = tf.train.AdamOptimizer().minimize(cost)


# Instantiate Session and Run 
sess = tf.Session()
sess.run(tf.global_variables_initializer())




















