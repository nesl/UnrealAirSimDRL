# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 01:43:40 2018

@author: natsn
"""

import sys
sys.path.append("D:\\Desktop\\Research\\Machine_Learning\\Anaconda\\Spyder\\Reinforcement_Learning_Master\\Util")
import tensorflow as tf
from CNNLayer import ConvPoolLayer
import numpy as np


# Assumtions: Pools are  placed at the back and then at a certain interval defined by pool size
# This class of code assumes 2D convolution operations and a slider in only the convolved directions
# Asumed to have a convolutional step of 1, as the conv modes are initialized to same as with most major algorithms
# Assumed that the pool size and the 
class VGGConvPoolBlock64():
    
    def __init__(self,  # Describes the filter sizes for the block
                 pool_flags = [False, True], relus_flags = [False, False],
                 batch_normalization_flags = [False, True],
                 kern_sizes = [[3,3,3,64],[3,3,64,64]],
                 conv_strides = [[1,1,1,1],[1,1,1,1]],
                 isBiaseds = [True, True],
                 conv_mode = ['SAME','SAME']): # The returned feature map will be of the same size as the inputted feature map
        
        # Setup Network Parameters
        self.kern_sizes = kern_sizes
        self.conv_strides = conv_strides
        self.isBiaseds = isBiaseds
        self.conv_mode = conv_mode
        self.layer_count = len(self.kern_sizes)
        self.batch_normalization_flags = batch_normalization_flags
        
        # Set up network 
        self.relu_flags = relus_flags # change to how many relus (self.layer_count is Maximum)
        self.pool_flags = pool_flags # change to how many pools (self.layer_count is Maximum)
        
        # Create List objects to hold the weight tensors
        self.conv_layers = []
    
    def set_layer_input_channel_dim(self, i):
        self.kern_sizes[0][2] = i
        # Instantiate tensors after input is set
        for ks,conv_stride,isBiased,conv_mode in zip(self.kern_sizes, self.conv_strides, self.isBiaseds, self.conv_mode):
            # The pool stride is set as default. 
            self.conv_layers.append(ConvPoolLayer(ks, conv_stride = conv_stride, isBiased = isBiased, conv_mode = conv_mode)) # Here the pool size must match the stride length
        
        for i in range(len(self.conv_layers)):
            print("Kernal Sizes: ", self.conv_layers[i].kern.shape) 
        
    # Returns the convolved, pooled, and potentially relu'd feature map
    def forward(self, Z):
        # Compute and return the convoled, pooled, and relued feature map
        for i in range(self.layer_count):
            pool_flag = self.pool_flags[i]
            relu_flag = self.relu_flags[i]
            Z = self.conv_layers[i].forward(Z, pool_flag, relu_flag)
            if self.batch_normalization_flags[i]:
                (mean, var) = tf.nn.moments(Z, axes = 0)
                Z = tf.nn.batch_normalization(Z, mean, var, offset = 0, scale = 1, variance_epsilon = 1e-8)
        return Z
    
    # For Counting operations with your pool
    def get_layer_output_channel_dim(self, kernal_layer):
        return self.conv_layers[kernal_layer].kern_size[3]
    def get_block_conv_stride(self):
        conv_stride_block_settings = self.conv_strides[0]
        return conv_stride_block_settings
    def get_block_pool_stride(self):
        pool_stride_block_settings = self.conv_layers[0].max_pool_stride 
        return pool_stride_block_settings
    def get_block_out_dim(self):
        return self.conv_layers[-1].kern_size[3] # Output channel
    def get_num_pools(self):
        return np.sum(np.array(self.pool_flags, dtype = np.int))


class VGGConvPoolBlock32(VGGConvPoolBlock64):
    
    def __init__(self, _pool_flags = [False, True], _relus_flags = [False, True],
                 _batch_normalization_flags = [False, True],
                 _kern_sizes = [[3,3,3,32],[3,3,32,32]],
                 _conv_strides = [[1,1,1,1],[1,1,1,1]],
                 _isBiaseds = [True, True],
                 _conv_mode = ['SAME','SAME']):
           VGGConvPoolBlock64.__init__(self, pool_flags = _pool_flags,
                                       relus_flags = _relus_flags,
                                       batch_normalization_flags = _batch_normalization_flags,
                                       kern_sizes = _kern_sizes,
                                       conv_strides = _conv_strides,
                                       isBiaseds = _isBiaseds,
                                       conv_mode = _conv_mode) # The returned feature map will be of the same size as the inputted feature map


class VGGConvPoolBlock128(VGGConvPoolBlock64):
    
    def __init__(self, _pool_flags = [False, True], _relus_flags = [False, True],
                 _batch_normalization_flags = [False, True],
                 _kern_sizes = [[3,3,3,128],[3,3,128,128]],
                 _conv_strides = [[1,1,1,1],[1,1,1,1]],
                 _isBiaseds = [True, True],
                 _conv_mode = ['SAME','SAME']):
           VGGConvPoolBlock64.__init__(self, pool_flags = _pool_flags,
                                       relus_flags = _relus_flags,
                                       batch_normalization_flags = _batch_normalization_flags,
                                       kern_sizes = _kern_sizes,
                                       conv_strides = _conv_strides,
                                       isBiaseds = _isBiaseds,
                                       conv_mode = _conv_mode) # The returned feature map will be of the same size as the inputted feature map
  

class VGGConvPoolBlock256(VGGConvPoolBlock64):
    
    def __init__(self, _pool_flags = [False, False ,True], _relus_flags = [False, False, True],
                 _batch_normalization_flags = [False, False, True],
                 _kern_sizes = [[3,3,3,256],[3,3,256,256],[3,3,256,256]],
                 _conv_strides = [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
                 _isBiaseds = [True, True, True],
                 _conv_mode = ['SAME','SAME','SAME']):
           VGGConvPoolBlock64.__init__(self, pool_flags = _pool_flags, 
                                       relus_flags = _relus_flags,
                                       batch_normalization_flags = _batch_normalization_flags,
                                       kern_sizes = _kern_sizes,
                                       conv_strides = _conv_strides,
                                       isBiaseds = _isBiaseds,
                                       conv_mode = _conv_mode) # The returned feature map will be of the same size as the inputted feature map


class VGGConvPoolBlock512(VGGConvPoolBlock64):
    
    def __init__(self, _pool_flags = [False, False ,True], _relus_flags = [False, False, True],
                 _batch_normalization_flags = [False, True, False],
                 _kern_sizes = [[3,3,3,512],[3,3,512,512],[3,3,512,512]],
                 _conv_strides = [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
                 _isBiaseds = [True, True, True],
                 _conv_mode = ['SAME','SAME','SAME']):
           VGGConvPoolBlock64.__init__(self, pool_flags = _pool_flags, 
                                       relus_flags = _relus_flags,
                                       batch_normalization_flags = _batch_normalization_flags,
                                       kern_sizes = _kern_sizes,
                                       conv_strides = _conv_strides,
                                       isBiaseds = _isBiaseds,
                                       conv_mode = _conv_mode) # The returned feature map will be of the same size as the inputted feature map




class ResNetBlock64(VGGConvPoolBlock64):
    def __init__(self, _pool_flags = [False, False], _relus_flags = [True, False],
                 _batch_normalization_flags = [True, True],
                 _kern_sizes = [[3,3,3,64],[3,3,64,64]],
                 _conv_strides = [[1,1,1,1],[1,1,1,1]],
                 _isBiaseds = [True, True],
                 _conv_mode = ['SAME','SAME']):
        VGGConvPoolBlock64.__init__(pool_flags = _pool_flags, 
                                    relus_flags = _relus_flags,
                                    batch_normalization_flags = _batch_normalization_flags,
                                 kern_sizes = _kern_sizes,
                                 conv_strides = _conv_strides,
                                 isBiaseds = _isBiaseds,
                                 conv_mode = _conv_mode)
    
    def forward(self, Z):
        # Compute and return the convoled, pooled, and relued feature map
        # No Pools should occur before addition -- will reduce the dimentionality
        X  = Z
        for i in range(self.layer_count):
            pool_flag = self.pool_flags[i]
            relu_flag = self.relu_flags[i]
            # Forwards
            Z = self.conv_layers[i].forward(Z, pool_flag, relu_flag)
            # Batch Normalization
            if self.batch_normalization_flags[i]:
                (mean, var) = tf.nn.moments(Z, axes = 0)
                Z = tf.nn.batch_normalization(Z, mean, var, offset = 0, scale = 1, variance_epsilon = 1e-8)
        # Add Residual
        Z = tf.add(Z, X) # In the residual network, we must add the input to the feature maps before a final relu 
        # Last Activation
        Z = tf.nn.relu(Z)
        return Z
        
        

class ResNetBlock128(ResNetBlock64):
    def __init__(self,  pool_flags = [False, False], relus_flags = [True, False],
                 batch_normalization_flags = [True, True],
                 kern_sizes = [[3,3,3,128],[3,3,128,128]],
                 conv_strides = [[1,1,1,1],[1,1,1,1]],
                 isBiaseds = [True, True],
                 conv_mode = ['SAME','SAME']):
        ResNetBlock64.__init__( _pool_flags = pool_flags, _relus_flags = relus_flags,
                 _batch_normalization_flags = batch_normalization_flags,
                 _kern_sizes = kern_sizes,
                 _conv_strides = conv_strides,
                 _isBiaseds = isBiaseds,
                 _conv_mode = conv_mode)


class ResNetBlock256(ResNetBlock64):
    def __init__(self, pool_flags = [False, False], relus_flags = [True, False],
                 batch_normalization_flags = [True, True],
                 kern_sizes = [[3,3,3,256],[3,3,256,256]],
                 conv_strides = [[1,1,1,1],[1,1,1,1]],
                 isBiaseds = [True, True],
                 conv_mode = ['SAME','SAME']):
        ResNetBlock64.__init__( _pool_flags = pool_flags, _relus_flags = relus_flags,
                 _batch_normalization_flags = batch_normalization_flags,
                 _kern_sizes = kern_sizes,
                 _conv_strides = conv_strides,
                 _isBiaseds = isBiaseds,
                 _conv_mode = conv_mode)




class ResNetBlock512(ResNetBlock64):
    def __init__(self, pool_flags = [False, False], relus_flags = [True, False],
                 batch_normalization_flags = [True, True],
                 kern_sizes = [[3,3,3,512],[3,3,512,512]],
                 conv_strides = [[1,1,1,1],[1,1,1,1]],
                 isBiaseds = [True, True],
                 conv_mode = ['SAME','SAME']):
        ResNetBlock64.__init__( _pool_flags = pool_flags, _relus_flags = relus_flags,
                 _batch_normalization_flags = batch_normalization_flags,
                 _kern_sizes = kern_sizes,
                 _conv_strides = conv_strides,
                 _isBiaseds = isBiaseds,
                 _conv_mode = conv_mode)














