#!/usr/bin/env python

from FullyConnectedLayer import FullyConnectedLayer
import tensorflow as tf

class FullyConnectedResNetBlock:
    # in_dim = The dimention of input
    # hl_dims = The dims of the hidden layers
    # batch norm will normalize vector using mean and variance
    def __init__(self, in_dim, hl_dims, bias = False, activation_func = tf.nn.relu, batch_normalization = True):
        self.in_dim = in_dim
        self.hl_dims = hl_dims
        self.bias = bias
        self.activation_func = activation_func
        self.batch_normalization = batch_normalization
        self.fc_layers = []
        # This must be equal, in order to add residual
        assert in_dim == hl_dims[-1]

        idim = self.in_dim
        for i in range(len(hl_dims)):
            if i == 0:  # Give Relu NN activation function
                self.fc_layers.append(FullyConnectedLayer(
                    idim, hl_dims[i], bias=self.bias, activation_fun=activation_func))
            else:  # Give No activation function
                self.fc_layers.append(FullyConnectedLayer(
                    idim, hl_dims[i], bias=self.bias, activation_fun=None))
            idim = hl_dims[i]

    def forward(self, Z):
            # Compute and return the convoled, pooled, and relued feature map
            # No Pools should occur before addition -- will reduce the dimentionality
            X = Z
            for i in range(len(self.fc_layers)):
                # Forwards
                Z = self.fc_layers[i].forward(Z)

                # Batch Normalization
                if self.batch_normalization:
                    (mean, var) = tf.nn.moments(Z, axes=0)
                    Z = tf.nn.batch_normalization(
                        Z, mean, var, offset=0, scale=1, variance_epsilon=1e-8)

            # Add Residual
            # In the residual network, we must add the input to the feature maps before a final relu
            Z = tf.add(Z, X)
            # Last Activation
            Z = tf.nn.relu(Z)
            return Z
