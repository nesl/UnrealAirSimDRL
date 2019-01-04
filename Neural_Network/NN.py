#!/usr/bin/env python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy
import copy
import sys
import os
from sklearn.preprocessing import LabelBinarizer
from FullyConnectedLayer import FullyConnectedLayer
from NNBlocks import FullyConnectedResNetBlock
sys.path.append("/home/natsubuntu/Desktop/UnrealAirSimDRL/Util")
import ExcelLoader as XL

# in_dim is networks input dimention
# out_dim is the networks output dimention
# hl_sizes is the networks hidden layer sizes
class ResidualNueralNetwork:
    def __init__(self, in_dim, hl_sizes, out_dim, isClassification = True, learning_rate = .0001):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, in_dim], name="X")
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, out_dim], name="Y")
        self.blocks = []
        self.learning_rate = learning_rate
        self.isClassification = isClassification

        # Create Layers / Blocks
        idim = in_dim
        self.blocks.append(FullyConnectedLayer(idim, hl_sizes[0])) # Non Res FC Layer
        idim=hl_sizes[0]
        self.blocks.append(FullyConnectedResNetBlock(idim, hl_sizes, bias = False)) # Res Block 
        self.blocks.append(FullyConnectedLayer(hl_sizes[-1], out_dim, activation_fun = None)) # Non Res Block
        

        # Roll On Through Those Tensors, Cowboy
        Z = self.X
        for i in range(len(self.blocks)):
            Z = self.blocks[i].forward(Z)
        
        self.Y_pred = Z
        # Now setup cost function
        if self.isClassification:
            self.Yk = tf.nn.softmax(self.Y_pred, axis=1)
            self.cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.Y, logits= self.Y_pred, dim=1))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            #self.optimizer = tf.train.AdagradDAOptimizer(.001).minimize(self.cost)
        else:  # Is Regressive
            self.cost = tf.reduce_sum(
                tf.squared_difference(self.Y_pred, self.Y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Start the session
        self.set_session(tf.Session())
        self.sess.run(tf.global_variables_initializer())
        print("Session Initialized: Network Params will be dumped to CNN_Parameters.txt")

    def set_session(self, session):
        self.sess = session

    # Pass in the full data chunk to train on
    def train(self, X, Y, mini_batch_chunk=2000, mini_batch_sz=64,
              mini_batch_ep=50, epochs=10, learning_rate=.001, reset_session=False):

        if reset_session:
            self.close_sess()  # reset

        training_examples = X.shape[0]
        batches = int(training_examples / mini_batch_chunk)
        print("Train start!")
        for e in range(epochs):
            for b in range(batches):
                # Grab New Batch
                rnd_indx = np.random.choice(len(X), mini_batch_chunk)
                X_mb = X[rnd_indx, :]
                Y_mb = Y[rnd_indx]
                for mb in range(mini_batch_ep):
                    rnd_indx = np.random.choice(
                        mini_batch_chunk, mini_batch_sz)
                    X_mbb = copy.deepcopy(np.array(X_mb[rnd_indx, :]))
                    Y_mbb = copy.deepcopy(np.array(Y_mb[rnd_indx]))
                    # Begin training:
                    loss, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
                                            self.X: X_mbb, self.Y: Y_mbb})
                    print("Training Epoch: ", e, ", Batch Chunk: ", b, "/", batches,
                          ", Round ", mb, "/", mini_batch_ep, ", Loss: ", loss)

    def predict(self, X, Y_target=None):
        if Y_target is None:
            if self.isClassification:
                return np.array(np.argmax(self.sess.run(self.Yk, feed_dict={self.X: X}), axis=1), dtype=np.float)
            else: # Regression
                return np.array(np.argmax(self.sess.run(self.Y_pred, feed_dict={self.X: X}), axis=1), dtype=np.float)
        else:  # Run prediction score as well
            Y_target = np.array(Y_target, dtype=np.float32)
            if self.isClassification:
                Y_op = np.array(np.argmax(self.sess.run(
                    self.Yk, feed_dict={self.X: X}), axis=1), dtype=np.float)
                percent_correct = np.sum(
                    np.array(Y_op == Y_target, dtype=np.int)) / Y_target.shape[0]
                return [percent_correct, Y_op]
            else: # Regression
                Y_op = np.array(np.argmax(self.sess.run(
                    self.Y_pred, feed_dict={self.X: X}), axis=1), dtype=np.float)
                percent_correct = np.sum(
                    np.array(Y_op == Y_target, dtype=np.int)) / Y_target.shape[0]
                return [percent_correct, Y_op]

    def save_tensor_weight(self,
                           directory=os.getcwd(),
                           step=100,
                           optional_string=""):
        saver = tf.train.Saver()
        saver.save(self.sess, directory + "CNN_Parameters" +
                   optional_string, global_step=step, write_meta_graph=False)

    def close_sess(self):
        self.sess.run(tf.global_variables_initializer())


### test the mnsit ################
# With FC ResNet, hits 96-98 %
def test_mnist():
    #Image Properties
    input = 784

    # Load in the CiFar-10 Dataset
    #(X,y) = XL.get_cifar10(data_set = 3)
    [X_train, y_train, X_test, y_test] = XL.get_mnist(norm=True, reshape_to_img = False)

    # Reshape the Y data
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    Y_test = lb.transform(y_test)

    # I/O Network Parameters
    idim = input
    odim = 10
    hl_sizes = [1000, 1000, 1000]

    # Initialize model
    model = ResidualNueralNetwork(idim, hl_sizes, odim)
    # Train Model
    model.train(X_train, Y_train, epochs = 10)

    [perc_correct_train, _] = model.predict(
        X_train[0:1000], y_train[0:1000])
    [perc_correct_test, y_guess] = model.predict(X_test[0:1000], y_test[0:1000])
    print("YTruth: ", y_test[0:15], "Estimates: ", y_guess[0:15])
    
    #Print Results
    print("Percent Correct for train set: ", perc_correct_train)
    print("Percent Correct for test set: ", perc_correct_test)
    #print("Predictions Train: ", ytr_dump)
    #print("Predictions Test: ", yt_dump)
    ### End Code

    # Plot and Visualize Dataset -- 10 digits
    for i in range(5):
        plt.figure(i)
        plt.title("Predicted: " + str(y_guess[i]) + ", Actual is: "  + str(y_test[i]))
        plt.imshow(np.reshape(X_test[i], (28, 28)), cmap='gray')
    plt.show()

    # Save Weights for Analysis
    model.save_tensor_weight(optional_string="Mnist_Dataset")
    model.close_sess()


if __name__ == "__main__":
    test_mnist()
