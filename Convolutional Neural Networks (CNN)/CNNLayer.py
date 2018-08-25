# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 01:40:56 2018

@author: natsn
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..\\Util")
import tensorflow as tf



# Convolutional and Max Pooling Layer
class ConvPoolLayer:
    # Kernal Size: NxMxWxZ in a tuple (NxM) - image, W = channel, Z = number of kernals youd like
    # Stride : Amount in increment we move between convolutions
    # Mode : dictates the dimentions of the feature map that we get back
    '''
    Computes a 2-D convolution given 4-D input and filter tensors.

    Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following:

    Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
    Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
    For each patch, right-multiplies the filter matrix and the image patch vector.
    '''
    # Kern Size should be filter width by filter height by numer of input channels by number of output channels
    def __init__(self, kern_size, conv_stride = [1,1,1,1], max_pool_size = [1,2,2,1],
                 isBiased = True, conv_mode = 'SAME'):
        
        self.kern_size = kern_size
        self.conv_stride = conv_stride
        self.max_pool_size = max_pool_size
        self.isBiased = isBiased
        self.conv_mode = conv_mode
        
        # Have the pool stride match the pool size
        self.max_pool_stride = max_pool_size
        OUTPUT_CHANNEL_DIM = 3

        # Kernal Matrix and Bias Vector -- Vector is broadcasted across all the output feature maps        
        self.kern = tf.Variable(tf.random_normal(shape = self.kern_size), dtype = tf.float32)
        self.bias = tf.Variable(tf.random_normal(shape = [self.kern_size[OUTPUT_CHANNEL_DIM]]), dtype = tf.float32)
        
        # Store Parameters in list
        self.params = [self.kern] + [self.bias]
        
    def forward(self, Z, pool = False, relu = False):
        # Compute and return the convolution
        # Z has the shape height by width by in channels by out channels
        print("Kern Shape: ", self.kern.shape, ", Input Shape: ", Z.shape)
        if self.isBiased:
            Z = tf.nn.bias_add(tf.nn.conv2d(Z,self.kern, self.conv_stride, padding = self.conv_mode), self.bias)
            if relu:
                Z = tf.nn.relu(Z)
            if pool:
                Z = tf.nn.max_pool(Z, self.max_pool_size, self.max_pool_stride, padding = "SAME")
            return Z
        else:
            Z = tf.nn.conv2d(Z, self.kern, self.conv_stride, padding = self.conv_mode)
            if relu:
                Z = tf.nn.relu(Z)
            if pool:
                Z = tf.nn.max_pool(Z, self.max_pool_size, self.max_pool_stride)
            return Z
