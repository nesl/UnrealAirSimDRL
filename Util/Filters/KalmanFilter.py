# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt



class DiscreteKalmanFilter:
    # input a discrete state space
    def __init__(self, ss, Q, V, x0, P0 = None):
        self.ss = ss
        self.A = np.matrix(ss.A)
        self.B = np.matrix(ss.B)
        self.C = np.matrix(ss.C)
        self.D = np.matrix(ss.D)
        self.x_bar = None
        self.x_hat = np.matrix(x0).reshape(len(x0),1)
        self.K = np.matrix(np.eye(self.A.shape[0])) # --unset
        self.M = np.zeros((self.A.shape[0], self.C.shape[1])) # Mt -- unset
        if P0 is None:
            self.P = np.matrix(np.eye(self.A.shape[0])) # P0
        else:
            self.P = P0
        self.Q = np.matrix(Q)
        self.V = np.matrix(V)
        self.I = np.matrix(np.eye(self.A.shape[0]))
        self.num_props = 1
        self.meas = None
        
    def propogate_state(self, U):
        if type(U) is list:
            tmp = self.x_hat
            self.num_props = len(U)
            for u in U:
                tmp = self.A*tmp + self.B*np.matrix(u).reshape(len(u),1)
            self.x_bar = tmp
        else:
            self.num_props = 1
            self.x_bar = self.A*self.x_hat + self.B*np.matrix(U).reshape(len(U),1)
        return self.x_bar
    def propogate_covariance(self):
        tmp = self.P
        for i in range(self.num_props):
            tmp = self.A*tmp*self.A + self.Q
        self.M = tmp
        return self.M
    def update_kalman_gain(self):
        self.K = self.M*self.C*(self.C*self.Q*self.C.T + self.V).I
        return self.K
    def set_measurement(self, measurement):
        assert self.meas.shape[0] == self.C.shape[0]
        self.meas = measurement
    def update_state(self):
        self.x_hat = self.x_bar + self.K*(self.meas - self.C*self.x_bar)
        return self.x_hat
    
    def update_covariance(self):
        self.P = (self.I - self.K*self.C)*self.M*(self.I - self.K*self.C).T + self.K*self.V*self.K.T 
        return self.P
    
    def run(self, steps, measurements, control_sequence, x_truth = None):
        assert steps == len(measurements) == len(control_sequence)
        for i in range(steps):
            self.propogate_state(control_sequence[i])
            self.propogate_covariance()
            self.update_kalman_gain()
            self.set_measurement(measurements[i])
            self.update_state()
            self.update_covariance()
    

class ContinousKalmanFilter:
    
    def __init__(self, A, B, C, D = None):
        pass
    def propogate_state(self):
        pass
    def propogate_covariance(self):
        pass
    def set_kalman_gain(self):
        pass
    def get_measurement(self):
        pass
    def update_state(self):
        pass
    def update_covariance(self):
        pass


# Note: The dt you use makes a big difference in terms of discrete time impulse response
def test_state_spaces():
    ssc = sig.StateSpace(Ac,Bc,Cc)
    dt = .01
    ssd = sig.cont2discrete((ssc.A,ssc.B,ssc.C,ssc.D), dt)
    t = np.arange(0,10,dt)
    
    plt.figure(1)
    plt.plot(t, np.array(sig.step((ssc.A,ssc.B,ssc.C,ssc.D), T = t)[1]).reshape(-1))
    plt.figure(2)
    plt.plot(t, np.array(sig.dstep(ssd, t = t)[1]).reshape(-1))
    plt.show()
    
    plt.figure(3)
    plt.plot(t, np.array(sig.impulse((ssc.A,ssc.B,ssc.C,ssc.D), T = t)[1]).reshape(-1))
    plt.figure(4)
    plt.plot(t, np.array(sig.dimpulse(ssd, t = t)[1]).reshape(-1))
    plt.show()


# The difference in amplitude for discrete and regular impulse is 10  
if __name__ == "__main__":
    Q = np.matrix(np.eye(3))*.1
    V = np.matrix(np.eye(3))*.2
    x0 = np.random.rand(3,1)
    P0 = np.matrix(np.eye(3))
    b = .3 # Ns/m
    m = 2 # kg
    gain = .1 # throttle model
    
    Ac = np.array([[0,1],[0, -b/m]])
    Bc = np.matrix([[0],[gain/m]])
    Cc = np.matrix([1,0])
    
    ssc = sig.StateSpace(Ac,Bc,Cc)
    dt = .2
    ssd = sig.cont2discrete((ssc.A,ssc.B,ssc.C,ssc.D), dt)
    t = np.arange(0,10,dt)
    
    kf = DiscreteKalmanFilter(ssd, Q, V, x0, P0)
    
    
    
    












