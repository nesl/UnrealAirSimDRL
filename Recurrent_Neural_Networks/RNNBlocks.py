# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:22:00 2018

@author: natsn
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class RNNBlock():

    def __init__(self, inp_sz, out_sz, num_hidden, num_lstmcells = 1):
        self.inp_sz = inp_sz
        self.num_hidden = num_hidden
        self.num_lstmcells = num_lstmcells
        self.weights = tf.Variable(tf.random_normal([num_hidden, out_sz]))
        self.biases = tf.Variable(tf.random_normal([out_sz]))

        self.params = [self.weights] + [self.biases]
    
    def forward(self, z, name=None):
        z = tf.reshape(z, [-1,self.inp_sz])
        z = tf.split(z, self.inp_sz,1)
        
        cell_list = []
        for i in range(self.num_lstmcells):
            cell_list.append(rnn.BasicLSTMCell(self.num_hidden, reuse = False, name = name))
        rnn_cell = rnn.MultiRNNCell(cell_list)

        outputs, states = rnn.static_rnn(rnn_cell, z, dtype=tf.float32)
        
        return tf.matmul(outputs[-1], self.weights) + self.biases















