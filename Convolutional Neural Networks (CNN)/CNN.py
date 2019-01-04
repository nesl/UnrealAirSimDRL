#!/usr/bin/env python

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import sys, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Get TF To use the GPU

sys.path.append("/home/natsubuntu/Desktop/UnrealAirSimDRL/Neural_Network")
sys.path.append("/home/natsubuntu/Desktop/UnrealAirSimDRL/Util")
import CNNBlocks
from FullyConnectedLayer import FullyConnectedLayer
import ExcelLoader as XL
import copy
import math


# CNN with ConvPool and FC layers ajustable
# Base layers are the VGG Structure
class CNN():
    
    # Input dimentions are the image tensors size
    # Output layer is the number of classes you wish to have
    # Number of conv layers is how many convolutional operations you want to run
    # Number fully connected layers is how many FC layers youd like the CNN to include
    # Kernal sizes for each convolutional layer
    def __init__(self, input_dims, output_dim,
                 fc_hl_sizes, isClassification = True, ):
        
        self.isClassification = isClassification
        self.output_dim = output_dim
        self.input_dims = input_dims
        
        # Implement VGG Blocks for a Mini - VGG Architecture
        # Begin Graph Abstraction
        self.CNN_Block_Layers = []
        self.FC_Layers = []
        x_dims = [None] + list(input_dims)
        
        # Make sure the user did not say the image dim is 2D -> 2D should be changes to at least 3d
        assert len(input_dims) == 3 # Should be inputted as a 3D image (if 2D -> third channel is 1)
        self.X = tf.placeholder(dtype = tf.float32, shape = x_dims, name = "X")
        self.Y = tf.placeholder(dtype = tf.float32, shape = [None, output_dim], name = "Y")
        
        # Holds the individual layer's pool size and pool stride settings
        self.pool_stride_block_settings = {'Num_Pools': [], 'Conv_Stride': [] ,'Pool_Stride': []}
        
        # OPTION 1:
        # Stack Convolutional Blocks -- VGG
        #ConvBlock = CNNBlocks.VGGConvPoolBlock64()
        #self.CNN_Block_Layers.append(ConvBlock)
        #ConvBlock = CNNBlocks.VGGConvPoolBlock64()
        #self.CNN_Block_Layers.append(ConvBlock)
        #ConvBlock = CNNBlocks.VGGConvPoolBlock128()
        #self.CNN_Block_Layers.append(ConvBlock)
        #ConvBlock = CNNBlocks.VGGConvPoolBlock256()
        #self.CNN_Block_Layers.append(ConvBlock)
        #ConvBlock = CNNBlocks.VGGConvPoolBlock512()
        #self.CNN_Block_Layers.append(ConvBlock)
        #ConvBlock = CNNBlocks.VGGConvPoolBlock512()
        #self.CNN_Block_Layers.append(ConvBlock)

        # OPTION 2: 
        # Stack Convolutional Blocks -- ResNet
        ConvBlock = CNNBlocks.VGGConvPoolBlock64()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.ResNetBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.ResNetBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.ResNetBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        ConvBlock = CNNBlocks.VGGConvPoolBlock128()
        self.CNN_Block_Layers.append(ConvBlock)
        
        # Determine Number of Stacked Convolutional Blocks
        self.num_conv_blocks = len(self.CNN_Block_Layers)
        # Initialize the Convolutional weights
        idim = self.input_dims[2]
        for i in range(self.num_conv_blocks):
            # Change the input of a block layer to have the same input channel dimention as the inputted image / feature map channel
            self.CNN_Block_Layers[i].set_layer_input_channel_dim(idim) #(0,idim) # Layer zero or 1
            idim = self.CNN_Block_Layers[i].get_layer_output_channel_dim() # Layer Zero or Layer 1 ( we choose layer one to connect to the next block)
            
            self.pool_stride_block_settings['Num_Pools'].append(self.CNN_Block_Layers[i].get_num_pools())
            self.pool_stride_block_settings['Pool_Stride'].append(self.CNN_Block_Layers[i].get_block_pool_stride())

            # Store Pool Stride information
            #self.pool_stride_block_settings['Num_Pools'].append(ConvBlock.get_num_pools())
            #self.pool_stride_block_settings['Pool_Stride'].append(ConvBlock.get_block_pool_stride())
           
        # Get input size for the fully connected layer
        FC_INPUT_SIZE = int(self.get_FC_input_size(input_dims,
                                               self.pool_stride_block_settings['Num_Pools'],
                                               self.pool_stride_block_settings['Pool_Stride']))
        idim = FC_INPUT_SIZE
        # Report FC Size for Conv - NN transition
        print("Fully Connected Input Size: ", idim)
        for i in range(len(fc_hl_sizes)):
            self.FC_Layers.append(FullyConnectedLayer(idim, fc_hl_sizes[i]))
            idim = fc_hl_sizes[i]
        self.FC_Layers.append(FullyConnectedLayer(idim, self.output_dim, activation_fun = None))
        
        # Rollout the network tensor abstraction
        Z = self.X
        # Convolutional Rollout
        for i in range(self.num_conv_blocks):
            Z = self.CNN_Block_Layers[i].forward(Z)
        
        # Reshape Z for the Fully Connected Rollout
        Z = tf.reshape(Z,(-1,FC_INPUT_SIZE))
        # Fully Conneccted Rollout
        for i in range(len(fc_hl_sizes)):
            Z = self.FC_Layers[i].forward(Z)
        
        self.Y_pred = self.FC_Layers[-1].forward(Z)
        
        # Here we either take the linear output unit, or we use the CNN for classification
        # If the classifcation flag is set to true, we use a softmax cross entropy with logits cost function
        if self.isClassification:
            self.Yk = tf.nn.softmax(self.Y_pred, axis = 1)
            self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.Y, logits = self.Y_pred, dim = 1))
            self.optimizer = tf.train.AdamOptimizer(.001).minimize(self.cost)
            #self.optimizer = tf.train.AdagradDAOptimizer(.001).minimize(self.cost)
        else: # Is Regressive
            self.cost = tf.reduce_sum(tf.squared_difference(self.Y_pred, self.Y))
            self.optimizer = tf.train.AdamOptimizer(.001).minimize(self.cost)
        

        # Start the session
        self.set_session(tf.Session())
        self.sess.run(tf.global_variables_initializer())
        print("Session Initialized: Network Params will be dumped to CNN_Parameters.txt")
        
    # We need to assume that our convolutions are 
    def get_FC_input_size(self, img_height_width, num_pools, Pool_Strides):
        
        # Initialize
        img_sizes = list(img_height_width)
        
        # This would have to change to deal with variable size and shaped tensors, but good for 2D Variable stride tensors
        for i in range(self.num_conv_blocks):
            # Reduce dimentionaility per the pool instructions
            if num_pools[i] != 0:
                if img_sizes[0] % (Pool_Strides[i][1] * num_pools[i]) != 0 or img_sizes[1] % (Pool_Strides[i][2] * num_pools[i]) != 0:
                    img_sizes[0] /= (Pool_Strides[i][1] * num_pools[i])
                    img_sizes[1] /= (Pool_Strides[i][2] * num_pools[i])
                    img_sizes[0] = math.ceil(img_sizes[0])
                    img_sizes[1] = math.ceil(img_sizes[1])
                else:
                    img_sizes[0] /= (Pool_Strides[i][1] * num_pools[i])
                    img_sizes[1] /= (Pool_Strides[i][2] * num_pools[i]) 
        return img_sizes[0] * img_sizes[1] * self.CNN_Block_Layers[-1].get_block_out_dim()
    
    def set_session(self, session):
        self.sess = session
    
    # Pass in the full data chunk to train on
    def train(self, X,Y, mini_batch_chunk = 2000, mini_batch_sz = 64, 
              mini_batch_ep = 50, epochs = 10, learning_rate = .001, reset_session = False):
        
        if reset_session:
            self.sess.run(tf.global_variables_initializer()) # reset
        
        training_examples = X.shape[0]
        batches = int(training_examples / mini_batch_chunk)
        print("Train start!")
        for e in range(epochs):
            for b in range(batches):
                # Grab New Batch
                rnd_indx = np.random.choice(len(X),mini_batch_chunk)
                X_mb = X[rnd_indx,:,:,:]
                Y_mb = Y[rnd_indx]
                for mb in range(mini_batch_ep):
                    rnd_indx = np.random.choice(mini_batch_chunk,mini_batch_sz)
                    X_mbb = copy.deepcopy(np.array(X_mb[rnd_indx,:,:,:]))
                    Y_mbb = copy.deepcopy(np.array(Y_mb[rnd_indx]))
                    # Begin training:
                    loss, _ = self.sess.run([self.cost, self.optimizer], feed_dict = {self.X: X_mbb, self.Y: Y_mbb} )
                    print("Training Epoch: ", e, ", Batch Chunk: ", b, "/", batches, 
                          ", Round ", mb, "/", mini_batch_ep, ", Loss: ", loss)
                    
    def predict(self, X, Y_target = None):
        Y_target = np.array(Y_target, dtype = np.float32)
        if Y_target is None:
            if self.isClassification:
                return np.array(np.argmax(self.sess.run(self.Yk, feed_dict = {self.X: X}),axis = 1), dtype = np.float)          
            else:
                return np.array(np.argmax(self.sess.run(self.Y_pred, feed_dict = {self.X: X}),axis = 1), dtype = np.float)
        else: # Run prediction score as well
            if self.isClassification:
                Y_op =  np.array(np.argmax(self.sess.run(self.Yk, feed_dict = {self.X: X}),axis = 1), dtype = np.float)
                percent_correct = np.sum(np.array(Y_op == Y_target, dtype = np.int)) / Y_target.shape[0]
                return [percent_correct, Y_op]
            else:
                Y_op = np.array(np.argmax(self.sess.run(self.Y_pred, feed_dict = {self.X: X}),axis = 1), dtype = np.float)
                percent_correct = np.sum(np.array(Y_op == Y_target, dtype = np.int)) / Y_target.shape[0]
                return [percent_correct, Y_op]
    
    def save_tensor_weight(self, 
                           directory = "/home/natsubuntu/Desktop",
                           step = 100,
                           optional_string = ""):
        saver = tf.train.Saver()
        saver.save(self.sess, directory + "CNN_Parameters" + optional_string, global_step = step, write_meta_graph = False)
    
    def close(self):
        self.sess.run(tf.global_variables_initializer())



### test the mnsit ################
def test_mnist():
    #Image Properties
    IM_WIDTH = 28
    IM_HEIGHT = 28
    IM_CHANNELS = 1
    
    # Load in the CiFar-10 Dataset
    #(X,y) = XL.get_cifar10(data_set = 3)
    [X_train, y_train, X_test, y_test] = XL.get_mnist(norm = True)
    
    # Reshape the Y data
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    Y_test = lb.transform(y_test)
    #plt.imshow(X[1], cmap = 'gray')
    
    # I/O Network Parameters
    idim = (IM_WIDTH, IM_HEIGHT, IM_CHANNELS)
    odim = 10
    hl_sizes = [1600,800,400] # Fully Connected layers. Conv Layers are setup in the object initializer
    
    # Initialize model
    model = CNN(idim, odim, hl_sizes)
    # Train Model
    model.train(X_train,Y_train)
    
    [perc_correct_train, ytr_dump] = model.predict(X_train[0:200,:,:,:], y_train[0:200])
    [perc_correct_test,yt_dump] = model.predict(X_test[0:200], y_test[0:200])
    
    # Save Weights for Analysis
    model.save_tensor_weight(optional_string = "Mnist_Dataset")
    model.close()
    
    #Print Results
    print("Percent Correct for train set: ", perc_correct_train)
    print("Percent Correct for test set: ", perc_correct_test)
    #print("Predictions Train: ", ytr_dump)
    #print("Predictions Test: ", yt_dump)
    ### End Code
    
    # Plot and Visualize Dataset
    for i in range(3):
        plt.figure(i)
        plt.title("Regular Image " + str(i) )
        plt.imshow(np.reshape(X_train[i], (28,28)), cmap = 'gray')
    plt.show()


if __name__ == "__main__":
    test_mnist()

    
    
    
    
    
    
    
    
    
    
    
    
    
    








