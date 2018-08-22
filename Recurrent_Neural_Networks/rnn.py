import tensorflow as tf
from tensorflow.python.ops import rnn

#from tensorflow.python.ops.rnn_cell import BasicRNNCell, GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import BasicRNNCell, MultiRNNCell

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs


# x: N*T*D array (batch_sz * sequence_length * features/input dimension)
def x2sequence(x, T, D, batch_sz):
    x = tf.transpose(x, (1,0,2))          #make it T-batchz_sz-features
    x = tf.reshape(x, (T*batch_sz, D))
    x = tf.split(x, T)
    return x
    
#learn the weight matrix
class HiddenLayer():
    def __init__(self, Mi, Mo):
        self.Mi = Mi
        self.Mo = Mo
        W = np.random.randn(Mi, Mo) / np.sqrt(Mi+Mo)
        b = np.zeros(Mo)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
        
    def get_hidden_layer_params(self):
        return self.params
    
    
    
class simpleRNN():
    def __init__(self, M):     
        self.M = M               #hidden layer size (num of layer units) 1 element
        
    def fit(self, X, Y, batch_sz =20, learning_rate=0.1, mu=0.9, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        
        N, T, D = X.shape # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        M = self.M
        self.f = activation
        
        hidden_layer = HiddenLayer(M, K)
        params = hidden_layer.get_hidden_layer_params()
        self.Wo = params[0]
        self.bo = params[1]
        
        # tf Graph input
        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

        # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
        sequenceX = x2sequence(tfX, T, D, batch_sz)     
    
        rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)
        outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))

        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, axis=1)
        targets = tf.reshape(tfY,(T*batch_sz,))     ####default -1
        
        #calculate the cost function
        cost_op = tf.reduce_mean(
                                 tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     logits = logits,
                                     labels = targets
                                                     )
                                 )

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)
        costs = []
        n_batches = N // batch_sz

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
                    # calculate c: 
                    _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c
                    for b in range(batch_sz):        
                        idx = (b+1)*T -1
                        n_correct += (p[idx] == Ybatch[b][-1])
                        
                    if i % 10 == 0:
                        print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                    if n_correct == N:
                        print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                        break
                    costs.append(cost)
                    
        if show_fig:
            plt.plot(costs)
            plt.show()



class multiRNN():
    def __init__(self, M):       #M here is a list
        self.M = M               #hidden layer size (num of layer units) list
        
    def fit(self, X, Y, batch_sz =20, learning_rate=0.1, mu=0.9, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        
        N, T, D = X.shape        # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        M = self.M
        self.f = activation
        
        hidden_layer = HiddenLayer(M[-1], K)
        params = hidden_layer.get_hidden_layer_params()
        self.Wo = params[0]
        self.bo = params[1]
        
        # tf Graph input
        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

        
        # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
        sequenceX = x2sequence(tfX, T, D, batch_sz)     
    
        cells = [BasicRNNCell(num_units=n, activation=self.f) for n in self.M]
        rnn_unit = MultiRNNCell(cells)
        outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M[-1]))

        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, axis=1)
        targets = tf.reshape(tfY,(T*batch_sz,))     
        
        #calculate the cost function
        cost_op = tf.reduce_mean(
                                 tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     logits = logits,
                                     labels = targets
                                                     )
                                 )

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)
        costs = []
        n_batches = N // batch_sz

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
                    # calculate c: 
                    _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c
                    for b in range(batch_sz):        
                        idx = (b+1)*T -1
                        n_correct += (p[idx] == Ybatch[b][-1])
                        
                    if i % 10 == 0:
                        print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                    if n_correct == N:
                        print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                        break
                    costs.append(cost)
                    
        if show_fig:
            plt.plot(costs)
            plt.show()







