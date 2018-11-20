# -*- coding: utf-8 -*-

#from filterpy.stats import Q_discrete_white_noise
from filterpy.common import Q_discrete_white_noise
from sympy import symbols, Matrix
import numpy as np
import math
import scipy.signal as sig
import matplotlib.pyplot as plt
import sympy as sym
import sympy
import os, sys
import time
sys.path.append("/home/natsubuntu/Desktop/UnrealAirSimDRL/Util/Control/")
from ControlLib import dlqr_finite_horizon, sim_dlqr, continous_time_least_norm_input
from sympy import (init_printing, Matrix, MatMul,integrate, symbols)
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class RadarStation(object):
    def __init__(self, pos, range_std, elev_angle_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_angle_std = elev_angle_std
    def reading_of(self, ac_pos):
        """ Returns (range, elevation angle) to aircraft.
        Elevation angle is in radians.
        """
        diff = np.subtract(ac_pos, self.pos)
        rng = np.linalg.norm(diff)
        brg = np.arctan2(diff[1], diff[0])
        return rng, brg

    def noisy_reading(self, ac_pos):

        """ Compute range and elevation angle to aircraft with
        simulated noise"""
        rng, brg = self.reading_of(ac_pos)
        rng += np.random.randn() * self.range_std
        brg += np.random.randn() * self.elev_angle_std
        return rng, brg

class PosSensor:
    def __init__(self, pos=(0, 0), vel=(0, 0), dt = 1, noise_std=1.):
        self.vel = vel
        self.dt = dt
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def sub_posxy(self, msg):
        print("Got Sensor Pos Update")
        self.vel = (msg.x, msg.y)
    def read(self, velx = None, vely = None):
        if velx == None:
            self.pos[0] += self.vel[0]*self.dt
        else:
            self.pos[0] += velx*self.dt
            self.vel[0] = velx
        if vely == None:
            self.pos[1] += self.vel[1]*self.dt
        else:
            self.pos[1] += vely*self.dt
            self.vel[1] = vely
        return (self.pos[0] + np.random.randn() * self.noise_std, self.pos[1] + np.random.randn() * self.noise_std)



class DiscreteKalmanFilter:
    # input a discrete state space
    def __init__(self, A,B,C, dt, Q, V, x0, P0=None, x0_truth = None):
        self.dt = dt
        self.A = A
        self.B = B
        self.C = C
        self.x_bar = x0
        self.x_hat = x0
        self.K = np.matrix(np.eye(self.A.shape[0])) # --unset
        self.M = np.zeros(self.A.shape) # Mt -- unset
        if P0 is None:
            self.P = np.matrix(np.zeros(self.A.shape)) # P0
        else:
            self.P = P0
        if x0_truth is None:
            self.x_truth = x0
        else:
            self.x_truth = x0_truth
        self.Q = np.matrix(Q)
        self.V = np.matrix(V)
        self.I = np.matrix(np.eye(self.A.shape[0]))
        self.msmt = 0
        self.xtruths = []
        self.us = []
        self.xbars = []
        self.xhats = []
        self.msmts = []
        
    def propogate(self, U):
        if type(U) is list:
            self.x_bar = self.x_hat
            self.M = self.P
            for u in U:
                self.x_bar = self.A*self.x_bar + self.B*u
                self.x_truth = self.A*self.x_truth + self.B*u
                self.M = self.A*self.M*self.A.T + self.Q
                self.xbars.append(self.x_bar)
                self.xhats.append(self.x_hat)
                self.xtruths.append(self.x_truth)
                self.us.append(u)
        else:
            self.x_bar = self.A*self.x_hat + self.B*U
            self.x_truth = self.A*self.x_truth + self.B*U
            self.M = self.A*self.M*self.A.T + self.Q
            self.xbars.append(self.x_bar)
            self.xtruths.append(self.x_truth)
            self.us.append(U)
        return self.x_bar

    def update_kalman_gain(self):
        self.K = self.M*self.C.T*(self.C*self.M*self.C.T + self.V).getI()
        return self.K
    def set_measurement(self, measurement = None):
        if measurement is None: # its a sim..so let based on filters properties
            noise = 0
            if self.V.shape[0] == 1:
                noise = np.random.normal(0,np.sqrt(np.asscalar(self.V)))
            else:
                noise = np.matrix(np.random.multivariate_normal(np.array(np.zeros((self.V.shape[0]))), np.sqrt(self.V))).T
            self.msmt = self.C *self.x_truth + noise
            #print(noise, self.x_truth, self.x_hat, self.C, self.msmt)
        else:
            assert measurement.shape[0] == self.C.shape[0]
            self.msmt = measurement
    def update_state(self):
        self.x_hat = self.x_bar + self.K*(self.msmt - self.C*self.x_bar)
        self.xhats.append(self.x_hat)
        return self.x_hat
    
    def update_covariance(self):
        self.P = (self.I - self.K*self.C)*self.M*(self.I - self.K*self.C).T + self.K*self.V*self.K.T 
        #self.P = (self.I - self.K*self.C)*self.M
        return self.P
    
    def update(self, measurement = None):
        self.update_kalman_gain()
        self.set_measurement(measurement)
        self.update_state()
        self.update_covariance()
        return self.x_hat, self.P

    def run_sim(self,control_sequence, plot = True):
        for i in range(len(control_sequence)):
            self.propogate(control_sequence[i])
            self.update()        
        if plot:
            for i in range(self.x_hat.shape[0]):
                plt.figure(i+1)
                print(self.xhats[0])
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                        [np.asscalar(x[i]) for x in self.xhats], color = 'blue')
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(xt[i]) for xt in self.xtruths], color = 'green')

                plt.title("DKF Reponse Y" + str(i) + " vs time")
            plt.show()

class ContinousKalmanFilter:
   # input a discrete state space
    def __init__(self, A, B, C, Q, V, x0, P0=None, x0_truth=None, tf =None):
        self.dt = .05
        self.A = A # F
        self.B = B # G
        self.C = C # H
        self.x_hat = x0
        self.dx_hat= None
        self.dx_truth = None
        self.dP = None
        self.K = np.matrix(np.eye(self.A.shape[0]))  # --unset
        if P0 is None:
            self.P = np.matrix(np.zeros(self.A.shape))  # P0
        else:
            self.P = P0
        if x0_truth is None:
            self.x_truth = x0
        else:
            self.x_truth = x0_truth
        self.Q = np.matrix(Q)
        self.V = V
        self.I = np.matrix(np.eye(self.A.shape[0]))
        self.msmt = 0
        self.xtruths = []
        self.us = []
        self.xhats = []
        self.msmts = []
        self.t = 0
        self.tf = tf

    def get_C(self):
        if callable(self.C):
            return self.C(self.tf, self.t)
        else:
            return self.C
    
    def get_V(self):
        if callable(self.V):
            return self.V(self.tf, self.t)
        else:
            return self.V

    def update_kalman_gain(self):
        self.K = self.P*self.get_C().T*self.get_V().I
        return self.K

    def set_measurement(self, measurement=None):
        if measurement is None:  # its a sim..so let based on filters properties
            noise = 0
            if self.V.shape[0] == 1:
                noise = np.random.normal(0, np.sqrt(np.asscalar(self.get_V())))
            else:
                noise = np.matrix(np.random.multivariate_normal(
                    np.array(np.zeros((self.V.shape[0]))), np.sqrt(self.get_V()))).T
            self.msmt = self.get_C() * self.x_truth + noise
            #print(noise, self.x_truth, self.x_hat, self.C, self.msmt)
        else:
            assert measurement.shape[0] == self.C.shape[0]
            self.msmt = measurement

    def update_state(self,u):
        self.dx_hat = self.A*self.x_hat + self.B*u + self.K*(self.msmt - self.get_C()*self.x_hat)
        self.x_hat =  self.x_hat + self.dx_hat * self.dt
        self.dx_truth = self.A*self.x_truth + self.B*u
        self.x_truth = self.x_truth + self.dx_truth * self.dt
        self.xhats.append(self.x_hat)
        self.xtruths.append(self.x_truth)
        self.t = self.t + self.dt
        return self.x_hat

    def update_covariance(self):
        self.dP = self.A*self.P + self.P*self.A.T - self.P*self.get_C().T*self.get_V().I*self.get_C()*self.P + self.B*self.Q*self.B.T
        self.P = self.P + self.dP * self.dt
        return self.P

    def update(self, u, measurement=None):
        self.update_kalman_gain()
        self.set_measurement(measurement)
        self.update_state(u)
        self.update_covariance()
        return self.x_hat, self.P

    def run_sim(self, control_sequence, plot=True):
        for i in range(len(control_sequence)):
            self.update(control_sequence[i])
        if plot:
            for i in range(self.x_hat.shape[0]):
                plt.figure(i+1)
                print(self.xhats[0])
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(x[i]) for x in self.xhats], color='blue')
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(xt[i]) for xt in self.xtruths], color='green')

                plt.title("CKF Reponse Y" + str(i) + " vs time")
            plt.show()

class DiscreteExtendedKalmanFilter:
    def __init__(self, x_sym, u_sym, f_xu, h_x, dt, 
                 Q, V, x0, P0=None, x0_truth=None, linearize_step=1):
        np.random.seed(np.random.choice([i for i in range(100)]))
        self.dt = dt
        self.linearize_step = linearize_step
        self.count = 0
        self.x_sym = x_sym
        self.u_sym = u_sym
        self.xu_sym = self.x_sym.col_join(self.u_sym)
        self.f = f_xu
        self.h = h_x
        
        """ self.F = self.f(x_sym, u_sym, dt, isSym = True).jacobian(self.x_sym)
        self.H = self.h(x_sym, dt, isSym = True).jacobian(self.x_sym)
        self.Vf = self.f(x_sym, u_sym, dt, isSym=True).jacobian(self.u_sym) # Derivative of f with respect to input u
         """
        self.F = sym.lambdify((x_sym,u_sym),self.f(x_sym, u_sym, dt, isSym = True).jacobian(self.x_sym))
        self.H = sym.lambdify([x_sym],self.h(x_sym, dt, isSym = True).jacobian(self.x_sym))
        self.Vf = sym.lambdify((x_sym, u_sym),self.f(x_sym, u_sym, dt, isSym=True).jacobian(self.u_sym)) # Derivative of f with respect to input u
        #print(self.F(x0,[.001,.002]))

        self.F_num = None
        self.H_num = None
        self.Vf_num = None
        self.x_bar = x0
        self.x_hat = x0
        self.K = np.matrix(np.eye(self.x_bar.shape[0]))  # --unset
        self.M = np.matrix(np.eye(self.x_bar.shape[0]))  # Mt -- unset
        if P0 is None:
            self.P = np.matrix(np.eye(self.x_bar.shape[0]))  # P0
        else:
            self.P = P0
        if x0_truth is None:
            self.x_truth = x0
        else:
            self.x_truth = x0_truth
        self.Q = np.matrix(Q)
        self.V = np.matrix(V)
        self.I = np.matrix(np.eye(self.x_bar.shape[0]))
        self.msmt = 0
        self.xtruths = []
        self.us = []
        self.xbars = []
        self.xhats = []
        self.msmts = []

    def numeric(self,f,xsym,xvals):
        tmp = f
        for i in range(len(xvals)):
            tmp = tmp.subs([(xsym[i],xvals[i])])
        return tmp
    
    def numeric_xu(self,F,x,u):
        return np.matrix(F(x, u), dtype=np.float)
    def numeric_x(self,H,x):
        return np.matrix(H(x), dtype=np.float)

    def propogate(self, U):
        if type(U) is list:
            for u in U:
                self.x_bar = self.f(self.x_hat,u)
                self.x_truth = self.f(self.x_truth, u)
                # Add Linearization
                self.F_num = self.numeric_xu(self.F, self.x_hat, u)
                self.Vf_num = self.numeric_xu(self.Vf, self.x_hat, u)

                # Propogate Covariance Matrices
                self.M = self.F_num*self.P*self.F_num.T + self.Q #self.Vf_num*self.Q*self.Vf_num.T
                self.xbars.append(self.x_bar)
                self.xhats.append(self.x_hat)
                self.xtruths.append(self.x_truth)
                self.us.append(u)
        else:
            self.count += 1
            U = U + .00005 # For numerical stability
            self.x_bar = self.f(self.x_hat, U, self.dt)
            self.x_truth = self.f(self.x_truth, U, self.dt)
            # Add Linearization
            if self.count % self.linearize_step == 0 or self.count == 1:
                self.F_num = self.numeric_xu(self.F, self.x_hat, U)
                self.Vf_num = self.numeric_xu(self.Vf, self.x_hat,U)
            # Propogate Covariance Matrices
            self.M = self.F_num*self.P*self.F_num.T + self.Q#self.Vf_num*self.Q*self.Vf_num.T
            self.xbars.append(self.x_bar)
            self.xtruths.append(self.x_truth)
            self.us.append(U)
        return self.x_bar

    def update_kalman_gain(self):
        if self.count % self.linearize_step == 0 or self.count == 1:
            self.H_num = self.numeric_x(self.H, self.x_bar)
            #self.H_num = np.matrix(self.numeric(self.H,self.x_sym,self.x_bar), dtype = np.float)
        self.K = self.M*self.H_num.T*(self.H_num*self.M*self.H_num.T + self.V).getI()
        return self.K

    def set_measurement(self, measurement=None):
        if measurement is None:  # its a sim..so based on filters and actual model
            noise = 0
            if self.V.shape[0] == 1:
                noise = np.random.normal(0, np.sqrt(np.asscalar(self.V)))
            else:
                noise = np.matrix(np.random.multivariate_normal(
                    np.array(np.zeros((self.V.shape[0]))), np.sqrt(self.V))).T
            self.msmt = self.h(self.x_truth, self.dt) + noise
            #print(noise, self.x_truth, self.x_hat, self.C, self.msmt)
        else:
            #assert measurement.shape[0] == self.H.shape[0]
            self.msmt = measurement

    def update_state(self):
        self.x_hat = self.x_bar + self.K*(self.msmt - self.h(self.x_bar, self.dt))
        self.xhats.append(self.x_hat)
        return self.x_hat

    def update_covariance(self):
        self.P = (self.I - self.K*self.H_num)*self.M*(self.I - self.K*self.H_num).T + self.K*self.V*self.K.T
        return self.P

    def update(self, measurement=None):
        self.update_kalman_gain()
        self.set_measurement(measurement)
        self.update_state()
        self.update_covariance()
        return self.x_hat, self.P

    def run_sim(self, control_sequence, plot=True):
        for i in range(len(control_sequence)):
            self.propogate(control_sequence[i])
            self.update()
        if plot:
            for i in range(self.x_hat.shape[0]):
                plt.figure(i+1)
                print(self.xhats[0])
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(x[i]) for x in self.xhats], color='blue')
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(xt[i]) for xt in self.xtruths], color='green')

                plt.title("Reponse Y" + str(i) + " vs time")
            plt.show()

class ContinousExtendedKalmanFilter:
    def __init__(self, x_sym, u_sym, f_xu, h_x, dt,
                 Q, V, x0, P0=None, x0_truth=None, linearize_step=1):
        np.random.seed(np.random.choice([i for i in range(100)]))
        self.dt = dt
        self.linearize_step = linearize_step
        self.count = 0
        self.x_sym = x_sym
        self.u_sym = u_sym
        self.xu_sym = self.x_sym.col_join(self.u_sym)
        self.f = f_xu
        self.h = h_x

        """ self.F = self.f(x_sym, u_sym, dt, isSym = True).jacobian(self.x_sym)
        self.H = self.h(x_sym, dt, isSym = True).jacobian(self.x_sym)
        self.Vf = self.f(x_sym, u_sym, dt, isSym=True).jacobian(
            self.u_sym) # Derivative of f with respect to input u
         """
        self.F = sym.lambdify((x_sym, u_sym), self.f(
            x_sym, u_sym, dt, isSym=True).jacobian(self.x_sym))
        self.H = sym.lambdify([x_sym], self.h(
            x_sym, dt, isSym=True).jacobian(self.x_sym))
        self.Vf = sym.lambdify((x_sym, u_sym), self.f(x_sym, u_sym, dt, isSym=True).jacobian(
            self.u_sym))  # Derivative of f with respect to input u
        #print(self.F(x0,[.001,.002]))

        self.F_num = None
        self.H_num = None
        self.Vf_num = None
        self.x_hat = x0
        self.dx_hat = None
        self.dx_truth = None
        self.dP = None
        self.K = np.matrix(np.eye(self.x_bar.shape[0]))  # --unset
        self.M = np.matrix(np.eye(self.x_bar.shape[0]))  # Mt -- unset
        if P0 is None:
            self.P = np.matrix(np.eye(self.x_bar.shape[0]))  # P0
        else:
            self.P = P0
        if x0_truth is None:
            self.x_truth = x0
        else:
            self.x_truth = x0_truth
        self.Q = np.matrix(Q)
        self.V = np.matrix(V)
        self.I = np.matrix(np.eye(self.x_bar.shape[0]))
        self.msmt = 0
        self.xtruths = []
        self.us = []
        self.xbars = []
        self.xhats = []
        self.msmts = []

    def get_C(self):
        if callable(self.C):
            return self.C(self.tf, self.t)
        else:
            return self.C

    def get_V(self):
        if callable(self.V):
            return self.V(self.tf, self.t)
        else:
            return self.V

    def numeric(self,f,xsym,xvals):
        tmp = f
        for i in range(len(xvals)):
            tmp = tmp.subs([(xsym[i],xvals[i])])
        return tmp
    
    def numeric_xu(self,F,x,u):
        return np.matrix(F(x, u), dtype=np.float)
    def numeric_x(self,H,x):
        return np.matrix(H(x), dtype=np.float)

    def propogate(self, U):
        if type(U) is list:
            for u in U:
                self.x_bar = self.f(self.x_hat,u)
                self.x_truth = self.f(self.x_truth, u)
                # Add Linearization
                self.F_num = self.numeric_xu(self.F, self.x_hat, u)
                self.Vf_num = self.numeric_xu(self.Vf, self.x_hat, u)

                # Propogate Covariance Matrices
                self.M = self.F_num*self.P*self.F_num.T + self.Q #self.Vf_num*self.Q*self.Vf_num.T
                self.xbars.append(self.x_bar)
                self.xhats.append(self.x_hat)
                self.xtruths.append(self.x_truth)
                self.us.append(u)
        else:
            self.count += 1
            U = U + .00005 # For numerical stability
            self.x_bar = self.f(self.x_hat, U, self.dt)
            self.x_truth = self.f(self.x_truth, U, self.dt)
            # Add Linearization
            if self.count % self.linearize_step == 0 or self.count == 1:
                self.F_num = self.numeric_xu(self.F, self.x_hat, U)
                self.Vf_num = self.numeric_xu(self.Vf, self.x_hat,U)
            # Propogate Covariance Matrices
            self.M = self.F_num*self.P*self.F_num.T + self.Q#self.Vf_num*self.Q*self.Vf_num.T
            self.xbars.append(self.x_bar)
            self.xtruths.append(self.x_truth)
            self.us.append(U)
        return self.x_bar

    def update_kalman_gain(self):
        if self.count % self.linearize_step == 0 or self.count == 1:
            self.H_num = self.numeric_x(self.H, self.x_bar)
            #self.H_num = np.matrix(self.numeric(self.H,self.x_sym,self.x_bar), dtype = np.float)
        self.K = self.M*self.H_num.T*(self.H_num*self.M*self.H_num.T + self.V).getI()
        return self.K

    def set_measurement(self, measurement=None):
        if measurement is None:  # its a sim..so based on filters and actual model
            noise = 0
            if self.V.shape[0] == 1:
                noise = np.random.normal(0, np.sqrt(np.asscalar(self.V)))
            else:
                noise = np.matrix(np.random.multivariate_normal(
                    np.array(np.zeros((self.V.shape[0]))), np.sqrt(self.V))).T
            self.msmt = self.h(self.x_truth, self.dt) + noise
            #print(noise, self.x_truth, self.x_hat, self.C, self.msmt)
        else:
            #assert measurement.shape[0] == self.H.shape[0]
            self.msmt = measurement

    def update_state(self):
        self.x_hat = self.x_bar + self.K*(self.msmt - self.h(self.x_bar, self.dt))
        self.xhats.append(self.x_hat)
        return self.x_hat

    def update_covariance(self):
        self.P = (self.I - self.K*self.H_num)*self.M*(self.I - self.K*self.H_num).T + self.K*self.V*self.K.T
        return self.P

    def update(self, measurement=None):
        self.update_kalman_gain()
        self.set_measurement(measurement)
        self.update_state()
        self.update_covariance()
        return self.x_hat, self.P

    def run_sim(self, control_sequence, plot=True):
        for i in range(len(control_sequence)):
            self.propogate(control_sequence[i])
            self.update()
        if plot:
            for i in range(self.x_hat.shape[0]):
                plt.figure(i+1)
                print(self.xhats[0])
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(x[i]) for x in self.xhats], color='blue')
                plt.plot([self.dt*j for j in range(len(self.xhats))],
                         [np.asscalar(xt[i]) for xt in self.xtruths], color='green')

                plt.title("Reponse Y" + str(i) + " vs time")
            plt.show()    

class TestUKF:
    def test_UKF():
        # Airplane Example
        init_printing(use_latex='mathjax')
        dt, phi = symbols('dt, phi')
        F_k = Matrix([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
        Q_c = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])*phi
        Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))
        # factor phi out of the matrix to make it more readable
        Q = Q / phi
        MatMul(Q, phi)

        pos1 = PosSensor(pos=(0, 0), vel=(2, 1), dt=1, noise_std=1)
        posxy = [pos1.read() for i in range(50)]
        plt.plot([x[0] for x in posxy], [y[1] for y in posxy])
        plt.show()

        dt = 3.  # 12 seconds between readings
        range_std = 5  # meters
        elevation_angle_std = math.radians(0.5)
        ac_pos = (0., 1000.)
        ac_vel = (100., 0.)
        radar_pos = (0., 0.)
        #h_radar.radar_pos = radar_pos
        #points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=0.)
        #kf = UKF(3, 2, dt, fx=f_radar, hx=h_radar, points=points)
        #kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)

    def normalize_angle(x):
        x = x % (2 * np.pi)
        if x > np.pi:
            x = x - 2 * np.pi
        return x

    def move(x, dt, u, wheelbase):
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt
        if abs(steering_angle) >= 0.001:  # is robot turning?
            beta = (dist / wheelbase) * np.tan(steering_angle)
            r = wheelbase / np.tan(steering_angle)  # radius
            sinh, sinhb = np.sin(hdg), np.sin(hdg + beta)
            cosh, coshb = np.cos(hdg), np.cos(hdg + beta)
            return x + np.array([-r*sinh + r*sinhb,
                                 r*cosh - r*coshb, beta])
        else:  # moving in straight line
            return x + np.array([dist*np.cos(hdg), dist*np.sin(hdg), 0])

    def residual_h(a, b):
        y = a - b
        # data in format [dist_1, bearing_1, dist_2, bearing_2,...]
        for i in range(0, len(y), 2):
            y[i + 1] = normalize_angle(y[i + 1])
        return y

    def residual_x(a, b):
        y = a - b
        y[2] = normalize_angle(y[2])
        return y

    def Hx(x, landmarks):
        """ takes a state variable and returns the measurement
        that would correspond to that state. """
        hx = []
        for lmark in landmarks:
            px, py = lmark
            dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
            angle = np.arctan2(py - x[1], px - x[0])
            hx.extend([dist, normalize_angle(angle - x[2])])
        return np.array(hx)

    def state_mean(sigmas, Wm):
        x = np.zeros(3)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
        x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
        x[2] = np.arctan2(sum_sin, sum_cos)
        return x

    def z_mean(sigmas, Wm):
        z_count = sigmas.shape[1]
        x = np.zeros(z_count)
        for z in range(0, z_count, 2):
            sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
            sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))
            x[z] = np.sum(np.dot(sigmas[:, z], Wm))
            x[z+1] = np.arctan2(sum_sin, sum_cos)
        return x

    def measure_move2(x, dt, isSym=False):
        landmark_pos = (5, 5)
        x = np.array(x).reshape(-1)
        if not isSym:
            px = landmark_pos[0]
            py = landmark_pos[1]
            dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
            hx = np.matrix([[dist],
                            [np.arctan2(py - x[1], px - x[0]) - x[2]]])
            return hx
        else:
            px = landmark_pos[0]
            py = landmark_pos[1]
            dist = sym.sqrt((px - x[0])**2 + (py - x[1])**2)
            hx = sym.Matrix([[dist],
                             [sym.atan2(py - x[1], px - x[0]) - x[2]]])
            return hx

    dt = 1.0
    wheelbase = 0.5

    def test_kalman_ukf():
        dt = 1.0
        wheelbase = 0.5
        landmarks = np.array([[5, 10], [10, 5], [15, 15]])
        cmds = [np.array([1.1, .01])] * 200
        ukf = run_localization(
            cmds, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
            sigma_range=0.3, sigma_bearing=0.1)
        print('Final P:', ukf.P.diagonal())

    def run_localization(
            cmds, landmarks, sigma_vel, sigma_steer, sigma_range,
            sigma_bearing, ellipse_step=1, step=10):

        plt.figure()
        points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0,
                                        subtract=residual_x)
        ukf = UKF(dim_x=3, dim_z=2*len(landmarks), fx=move, hx=Hx,
                  dt=dt, points=points, x_mean_fn=state_mean,
                  z_mean_fn=z_mean, residual_x=residual_x,
                  residual_z=residual_h)
        ukf.x = np.array([2, 6, .3])
        ukf.P = np.diag([.1, .1, .05])
        ukf.R = np.diag([sigma_range**2,
                         sigma_bearing**2]*len(landmarks))
        ukf.Q = np.eye(3)*0.0001
        sim_pos = ukf.x.copy()
        # plot landmarks
        if len(landmarks) > 0:
            plt.scatter(landmarks[:, 0], landmarks[:, 1],
                        marker='s', s=60)
        track = []
        for i, u in enumerate(cmds):
            sim_pos = move(sim_pos, dt/step, u, wheelbase)
            track.append(sim_pos)
            if i % step == 0:
                ukf.predict(u=u, wheelbase=wheelbase)
                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                        facecolor='k', alpha=0.3)
                x, y = sim_pos[0], sim_pos[1]
                z = []
                for lmark in landmarks:
                    dx, dy = lmark[0] - x, lmark[1] - y
                    d = np.sqrt(dx**2 + dy**2) + np.random.randn()*sigma_range
                    bearing = np.arctan2(lmark[1] - y, lmark[0] - x)
                    a = (normalize_angle(bearing - sim_pos[2] +
                                         np.random.randn()*sigma_bearing))
                    z.extend([d, a])
                ukf.update(z, landmarks=landmarks)
                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                        facecolor='g', alpha=0.8)
        track = np.array(track)
        plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
        plt.axis('equal')
        plt.title("UKF Robot localization")
        plt.show()
        return ukf

class ParticleFilter:
    def __init__(self):
        pass


# This Class Solves the Deterministic Time Percect Information LQ Control Problem
class ContinousTimeLinearQuadraticRegulator_FiniteHorizon:
    def __init__(self, A, G, Q, R, x0, Qf, tf=10):
        self.dt = .01  # Propogation Interval for the c-time dynamics
        self.A = A  # Dynamics Matrix
        self.G = G  # Control Gain
        self.Q = Q  # This is the weighting across the time process
        self.R = R  # This is the input enrgy wieghting
        self.Qf = Qf  # This is the final cost condition on the cost equation
        self.tf = tf
        self.x0 = x0

        # If the horizon is finite we need to:
        # 1) Setup Boundary COnditions: Sf = Qf, PIf = 0
        # Use the resulting minimization of the hamilton jacobi bellman dynamic for stochastic programming
        # We know Optimal control law, need to store Propogation of Sf backwards in time
        # At each propogation backwards with the riccatti expression, we store Sn-1 and
        self.St = Qf  # Initially set the boundary condition of our energy term to roll back from the final state weighting
        self.neg_dSt = 0  # change in St rolled back thru time
        self.St0 = 0  # Final energy expression at initial time after backwards propogation
        self.lambdas = []
        self.times = np.arange(0, self.tf, self.dt)
        self.Sts = []
        self.xts = []
        #self.solve_control()
    # Finite_horizon

    def solve_control(self):
        for t in self.times:
            self.neg_dSt = self.St*self.A + self.A.T*self.St - self.St*self.G*self.R.I*self.G.T*self.St + self.Q
            self.St = self.St + self.neg_dSt * self.dt
            self.Sts.append(self.St)
        self.S0 = self.St
        self.Sts = [s for s in reversed(self.Sts)]
        self.lambdas = [self.R.I*self.G.T*St for St in self.Sts]
        self.cost = self.x0.T*self.St*self.x0

    def plot_control_gains(self):
        time = np.arange(0, self.tf, self.dt)
        time_to_go = [j for j in reversed(time)]
        for i in range(self.lambdas[0].shape[1]):
            plt.plot(time, [l[0, i] for l in reversed(self.lambdas)])
        plt.title("Gains vs Time-To-Go")
        plt.gca().invert_xaxis()
        plt.show()
    
    # Plot The Energy St
    def plot_ricatti(self):
        plt.close()
        time = np.arange(0, self.tf, self.dt)
        for i in range(self.Sts[0].shape[0]):
            plt.plot(time, [s[i, i] for s in self.Sts])
        plt.title("CLQ Determ Ricatti Energy vs Time")
        #plt.gca().invert_xaxis()
        plt.show()
    
    def run_sim(self, plot=True):
        self.solve_control()
        plt.figure()
        xt = self.x0
        for i in range(len(self.times)):
            self.xts.append(xt)
            dxt = (self.A - self.G*self.lambdas[i])*xt
            xt = xt + dxt*self.dt

        for i in range(self.xts[0].shape[0]):
            plt.plot(self.times, [x[i, 0] for x in self.xts])
            plt.title("C-Time LQ Deterministic Reponse X" + str(i))
            plt.ylabel("Response")
            plt.xlabel("Time")

        plt.show()
        print("Cost: ", self.cost)
        return self.cost

# This Class Solves the Deterministic LQ Problem using Discrete Lyapunov Equations 
class DiscreteTimeLinearQuadraticRegulator_FiniteHorizon:
    def __init__(self, A,G,Q,R,x0,Qf,dt = .05,tf = 10):
        self.dt = dt  # Propogation Interval for the c-time dynamics
        self.A = A  # Dynamics Matrix
        self.G = G  # Control Gain
        self.Q = Q  # This is the weighting across the time process
        self.R = R  # This is the input energy wieghting
        self.Qf = Qf  # This is the final cost condition on the cost equation
        self.tf = tf # Final Time
        self.x0 = x0 # Initial State

        # If the horizon is finite we need to:
        # Setup Boundary Conditions: Sf = Qf, PIf = 0
        # We know Optimal control law, need to store Propogation of Sf backwards in time
        # At each propogation backwards with the riccatti expression, we store Sn-1 and
        self.St = Qf  # Initially set the boundary condition of our energy term to roll back from the final state weighting
        self.St_prev = 0  # change in St rolled back thru time
        self.St0 = 0  # Final energy expression at initial time after backwards propogation
        self.lambdas = []
        self.times = np.arange(0, self.tf, dt)
        self.Sts = []
        self.xts = []
        #self.solve_control()

    # Plot Gain Matrix
    def plot_control_gains(self):
        time = np.arange(0, self.tf, self.dt)
        time_to_go = [j for j in reversed(time)]
        print(time_to_go)
        for i in range(self.lambdas[0].shape[1]):
            plt.plot(time[0:-1], [l[0, i] for l in reversed(self.lambdas)])
        plt.title("DKF Deterministic Gains vs Time-To-Go")
        plt.gca().invert_xaxis()
        plt.show()

    # Plot The Energy St
    def plot_ricatti(self):
        plt.close()
        time = np.arange(0, self.tf, self.dt)
        for i in range(self.Sts[0].shape[0]):
            plt.plot(time, [s[i, i] for s in self.Sts])
        plt.title("DLQ Deterministic Param Ricatti Energy vs Time")
        #plt.gca().invert_xaxis()
        plt.show()

    # Finite_horizon dynamic programming
    def solve_control(self, tf = None):
        if tf is not None:
            self.tf = tf
            self.times = np.arange(0, self.tf, self.dt)
        for t in self.times:
            self.Sts.append(self.St)
            self.St_prev = self.Q + self.A.T*self.St*self.A - (self.G.T*self.St*self.A).T*(self.R + self.G.T*self.St*self.G).I*(self.G.T*self.St*self.A)
            self.St = self.St_prev
        
        self.S0 = self.St
        St = [i for i in reversed(self.Sts)]
        self.lambdas = [(self.R + self.G.T*St[i+1]*self.G).I*(self.G.T*St[i+1]*self.A) for i in range(len(St)-1)]
        self.cost = self.x0.T*self.St*self.x0

    # Finite_horizon dynamic programming
    def solve_control_non_linear(self,ekf,x_ref,linref,tf=None):
        xt = x_ref
        u = np.matrix([1, np.arctan2(x_ref[1], x_ref[0])]).T
        A = ekf.numeric_xu(ekf.F,linref, u)
        G = ekf.numeric_xu(ekf.Vf, linref, u)
        self.lambdas = []
        if tf is not None:
            self.tf = tf
            self.times = np.arange(0, self.tf+ self.dt, self.dt)
        for t in self.times:
            self.Sts.append(self.St)
            self.St_prev = self.Q + A.T*self.St*A - \
                (G.T*self.St*A).T*(self.R + G.T *
                                             self.St*G).I*(G.T*self.St*A)
            # Propogate the non-linear states backwards
            lam = (self.R + G.T*self.St*G).I *(G.T*self.St*A)
            self.St = self.St_prev
            #np.linalg.matrix_rank(A)
            self.lambdas.append(lam)
            #xt = (A-G*lam).I*xt
            #u = -lam*xt # optimal control for this last state
            #A = ekf.numeric_xu(ekf.F, xt, u)
            #G = ekf.numeric_xu(ekf.V, x_ref, u)

        self.S0 = self.St
        St = [i for i in reversed(self.Sts)]
        self.lambdas = [lam for lam in reversed(self.lambdas)]
        self.cost = self.x0.T*self.St*self.x0

    
    def run_sim(self, plot=True):
        self.solve_control()
        plt.figure(1)
        xt = self.x0
        for i in range(len(self.times)-1):
            self.xts.append(xt)
            xt = (self.A - self.G*self.lambdas[i])*xt

        for i in range(self.xts[0].shape[0]):
            plt.plot(self.times[0:-1], [x[i, 0] for x in self.xts])
            plt.title("D-Time LQ Deterministic Reponse X" + str(i))
        plt.ylabel("Response")
        plt.xlabel("Time")
        plt.show()
        print("Cost: ", self.cost)
        return self.cost

class DiscreteTimeLinearQuadraticRegulator_ParamVar_FiniteHorizon:
    def __init__(self, A, G, Q, R, x0, P0, Qf,Saa,Sbb,Sab, dt=.01, tf=10):
        self.dt = dt  # Propogation Interval for the c-time dynamics
        self.A = A  # Dynamics Matrix
        self.G = G  # Control Gain
        self.Q = Q  # This is the weighting across the time process
        self.R = R  # This is the input enrgy wieghting
        self.Qf = Qf  # This is the final cost condition on the cost equation
        self.tf = tf
        self.x0 = x0
        self.Saa = Saa
        self.Sbb = Sbb
        self.Sab = Sab
        self.P0 = P0
        # If the horizon is finite we need to:
        # 1) Setup Boundary COnditions: Sf = Qf, PIf = 0
        # Use the resulting minimization of the hamilton jacobi bellman dynamic for stochastic programming
        # We know Optimal control law, need to store Propogation of Sf backwards in time
        # At each propogation backwards with the riccatti expression, we store Sn-1 and
        self.St = Qf  # Initially set the boundary condition of our energy term to roll back from the final state weighting
        self.St_m  = Qf
        self.prev_St = 0  # change in St rolled back thru time
        self.St0 = 0  # Final energy expression at initial time after backwards propogation
        self.lambdas = []
        self.times = np.arange(0, tf, dt)
        self.Sts = []
        self.Sts_m = [] 
        self.xts = []
        #self.m = (self.Saa*self.Sbb + self.Saa*self.G*self.G.T + self.Sbb*self.A*self.A.T - self.Sab.T*self.Sab - 2*self.A*self.G*self.Sab.T)*(self.Sbb + self.G*self.G.T).I
        self.m = self.Saa + self.A.T*self.A - (self.Sab + self.A*self.G).T*(self.Sbb + self.G*self.G.T).I*(self.Sab + self.A*self.G)
        print(self.m)
        self.solve_control()

    # Finite_horizon
    def solve_control(self):
        for t in self.times:
            self.Sts.append(self.St)
            self.Sts_m.append(self.St_m)
            #self.prev_St = self.Q + self.A.T*self.St*self.A + self.Saa*self.St - \
            #    (self.G.T*self.St*self.A + self.Sab*self.St).T*(self.R + self.G.T * self.St*self.G + self.Sbb*self.St).I*(self.G.T*self.St*self.A + self.Sab*self.St)
            self.prev_St = self.Q + self.A.T*self.St*self.A + self.Saa*self.St - \
                (self.St*self.St*(self.Sab + self.A*self.G)
                 * (self.Sab + self.A*self.G))*(self.R + self.St*(self.Sbb + self.G*self.G)).I

            self.St_m = self.St
            self.St_m = self.m*self.St_m
            self.St = self.prev_St

        self.S0 = self.St
        #self.alpha0 = self.alpha_t
        St = [i for i in reversed(self.Sts)]
        self.lambdas = [(self.R + self.G.T*St[i+1]*self.G + self.Sbb*St[i+1]).I *
                        (self.G.T*St[i+1]*self.A + self.Sab*St[i+1]) for i in range(len(St)-1)]
        self.cost = self.x0.T*self.S0*self.x0

   # Plot Gain Matrix
    def plot_control_gains(self):
        plt.close()
        time = np.arange(0, self.tf, self.dt)
        time_to_go = [j for j in reversed(time)]
        print(time_to_go)
        for i in range(self.lambdas[0].shape[1]):
            plt.plot(time[0:-1], [l[0, i] for l in reversed(self.lambdas)])
        plt.title("DLQ w/ Param-Var Gains vs Time-To-Go")
        plt.gca().invert_xaxis()
        #plt.show()
        return "LQ Gains @ Saa: " + str(self.Saa[0,0]) + ", Sab: " + str(self.Sab[0,0]) + ", Sbb: " + str(self.Sbb[0,0])


    def plot_covariance(self):
        times = np.arange(0, self.tf, self.dt)
        cov_prop_const = self.Saa - (self.Sbb + self.G*self.G.T).I*(2*(self.Sab + self.A*self.G)*(
            self.Sbb + self.G*self.G.T) + self.Sbb*(self.Sab + self.A*self.G)*(self.Sab + self.A*self.G)).T*(self.Sbb + self.G*self.G.T).I
        cov = self.P0 
        covs = []
        xt = self.x0
        for i in range(len(self.times)-1):
            covs.append(cov)
            cov = self.m*cov + xt.T*cov_prop_const*xt
            xt = (self.A - self.G*self.lambdas[i])*xt
        for i in range(covs[0].shape[0]):
            plt.plot(times[0:-1], [cov[i, i] for cov in covs])
        plt.title("DLQ Determ Param Covariance vs Time")
        return "Cov for M: " + str(round(self.m[i,i],4))     

    def plot_ricatti_with_m(self):
        times = np.arange(0, self.tf, self.dt)
        for i in range(self.Sts_m[0].shape[0]):
            plt.plot(times, [s[i,i] for s in reversed(self.Sts_m)])
        plt.title("DLQ Determ Param Ricatti with M vs Time")
        return "M: " + str(round(self.m[i, i], 4))
    # Plot The Energy St 
    def plot_ricatti(self):
        time = np.arange(0, self.tf, self.dt)
        for i in range(self.Sts[0].shape[0]):
            plt.plot(time, [s[i, i] for s in reversed(self.Sts)])
        return "Saa: " + str(self.Saa[0,0]) + ", Sab: " + str(self.Sab[0,0]) + ", Sbb: " + str(self.Sbb[0,0])

        plt.title("DLQ Determ Param Ricatti Energy vs Time")
        #plt.gca().invert_xaxis()
        #plt.show()

    # Run Simulator
    def run_sim(self, plot=True):
        plt.figure(1)
        xt = self.x0
        for i in range(len(self.times)-1):
            self.xts.append(xt)
            xt = (self.A - self.G*self.lambdas[i])*xt

        for i in range(self.xts[0].shape[0]):
            plt.plot(self.times[0:-1], [x[i, 0] for x in self.xts])

        plt.title("DLQ Determ w/ Param Var Reponse X vs time")
        plt.show()
        print("Cost: ", self.cost)
        return "Response w/ m=" + str(round(self.m[i,i],4))

# The Non-Perfect Measurement Case: The Linear Quadratic Gaussians
# Assumes a Finite Horizon -- The cascaded LQR and Kalman FIlter using recursive dynamic programming and Lyapunov/Riccatti Expressions
class DiscreteTimeLinearQuadraticGaussian:
    def __init__(self,F,G,Q,Qf,R,H,V,Qproc,P0,x0,x0_truth=None,dt=.05):
        self.dlqr = DiscreteTimeLinearQuadraticRegulator_FiniteHorizon(
            F, G, Q, R, x0, Qf, dt)
        self.dkf = DiscreteKalmanFilter(F,G,H,dt,Qproc,V,x0,P0,x0_truth)
        self.state = x0
        self.states = []
    def track_ref(self,xref_val, duration, plot = False):
        self.dlqr.solve_control(tf= duration)
        self.state[-1] = xref_val
        self.dkf.x_hat = self.state
        for i in range(len(self.dlqr.times)-1):
            self.states.append(self.state)
            self.state = self.dkf.propogate(-self.dlqr.lambdas[i]*self.state)
            self.state, _ = self.dkf.update()
        if plot:
            self.plot_trajectory()
    def plot_trajectory(self):
        times = np.arange(0,len(self.states)*self.dlqr.dt,self.dlqr.dt)
        for i in range(self.states[0].shape[0]):
            plt.plot(times, [x[i, 0] for x in self.states])
        plt.title("LQG Deterministic Reponse X")
        plt.ylabel("Response")
        plt.xlabel("Time")
        plt.show()

class DiscreteTimeExtendedLinearQuadraticGaussian:
    def __init__(self, f_xu, h_x, Q, R, Qf, x_sym, u_sym,
                 Q_noise, V, x0, dt, P0=None, x0_truth=None,linearize_step =1):
        self.dlqr = DiscreteTimeLinearQuadraticRegulator_FiniteHorizon(None,None,Q,R,x0,Qf,dt)
        self.dekf = DiscreteExtendedKalmanFilter(x_sym,u_sym,f_xu,h_x,dt,Q_noise,V,x0,P0,x0_truth,linearize_step=1)
        self.state = x0
        self.states = []
    
    # Make sure your Q,Qf and Rs make sense
    def track_references(self,xrefs, tf, plot = True):
        self.state = self.dekf.x_hat
        self.state[4] = xrefs[0]
        self.state[6] = xrefs[1]
        self.state[8] = xrefs[2]
        
        ref = np.matrix([xrefs[0],xrefs[1],xrefs[2],0,xrefs[0],0,xrefs[1],0,xrefs[2]]).T
        linref = (ref + self.state) / 2
        self.dlqr.solve_control_non_linear(self.dekf,ref,linref,tf=tf)

        for i in range(len(self.dlqr.times)-1):
            self.states.append(self.state)
            print(self.state, -self.dlqr.lambdas[i]*self.state, self.dekf.x_truth)
            self.state = self.dekf.propogate(-self.dlqr.lambdas[i]*self.state)
            self.state, _ = self.dekf.update()
        if plot:
            for i in range(self.states[0].shape[0]):
                plt.plot(self.dlqr.times[0:-1], [x[i, 0] for x in self.states])
            plt.show()
            plt.title("D-Time Extended LQG Reponse X" + str(i))
            plt.ylabel("Response")
            plt.xlabel("Time")
            plt.show()
# Assumes a Finite Horizon -- The cascaded LQR and Kalman Filter using Hamilton-Jacobi-Bellman Analysis
class ContinousTimeLinearQuadraticGaussian:
    def __init__(self, F, G, Q, Qf, R, H, V, Qproc, P0, x0, x0_truth=None, dt=.05):
        self.dlqr = DiscreteTimeDeterministicLinearQuadraticRegulator_FiniteHorizon(
            F, G, Q, R, x0, Qf, dt)
        self.dkf = DiscreteKalmanFilter(
            F, G, H, dt, Qproc, V, x0, P0, x0_truth)
        self.state = x0
        self.states = []

    def track_ref(self, xref_val, duration, plot=False):
        self.dlqr.solve_control(tf=duration)
        self.state[-1] = xref_val
        for i in range(len(self.dlqr.times)):
            self.states.append(self.state)
            self.state = self.dkf.propogate(-self.dlqr.lambdas[i]*self.state)
            self.state = self.dkf.update()
        if plot:
            self.plot_trajectory()

    def plot_trajectory(self):
        times = np.arange(0, len(self.states)*self.dt, self.dt)
        for i in range(self.states[0].shape[0]):
            plt.plot(self.times[0:-1], [x[i, 0] for x in self.states])
        plt.title("LQG Deterministic Reponse X")
        plt.ylabel("Response")
        plt.xlabel("Time")
        plt.show()


class ACSim:
    def __init__(self, pos, vel, vel_std):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.vel_std = vel_std

    def update(self, dt):
        """ Compute and returns next position. Incorporates
        random variation in velocity. """
        dx = self.vel*dt + (np.random.randn() * self.vel_std) * dt
        self.pos += dx
        return self.pos

def Vt(t,tf):
    R1 = 15e-6
    R2 = 1.67e-3
    return np.matrix([R1 + R2/(tf - t)**2])

def Ht(t,tf):
    Vc = 300
    return np.matrix([1/Vc*(tf - t), 0, 0])


def test_lq_stochastic_ctime():
    tau = 2
    t0 = 0
    tf = 10
    b = 1.52e-2
    x0 = np.matrix([2,8,15]).T
    x0_truth = np.matrix([0, 0, 0]).T
    A = np.matrix([[0,1,0],
                    [0,0,-1],
                    [0,0,-1/tau]])
    G = np.matrix([0,1,0]).T
    C = H
    B = np.matrix([0,0,1]).T
    W = 10000
    Q = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    R = np.matrix([b])
    Qf = np.matrix([[1,0,0],[0,0,0],[0,0,0]])
    Vc = V
    P0 = np.matrix([[0,0,0],[0,20,0],[0,0,10000]])
    lqr = ContinousTimeLinearQuadraticRegulator_FiniteHorizon(A,G,Q,R,x0,Qf,tf)
    lqr_stoch = ContinousTimeLinearQuadraticRegulator_FiniteHorizon(A,G,B,Q,R,W,x0,Qf,P0,tf)
    #ckf = ContinousKalmanFilter(A, G, C, B*W*B.T, Vc, x0, P0, x0_truth, tf = tf)
    
    lqr.run_sim()
    #lqr_stoch.run_sim()
    #ckf.run_sim()

    lqr.plot_control_gains()
    #lqr_stoch.plot_control_gains()

    lqr.plot_ricatti()
    #Test Discrete Time Response
    dt = .01
    ssd = sig.cont2discrete((A,B,np.matrix([1,0,0]), np.matrix([0])) ,dt)
    Ad = ssd[0]
    Gd = ssd[1]
    Cd = V
    Dd = ssd[3]
    Wd = Ad.T*B*W*B.T*Ad
    dlqr = DiscreteTimeDeterministicLinearQuadraticRegulator_FiniteHorizon(Ad, Gd, Q, R, x0, Qf, dt, tf = 10)
    dlqr_stoch = DiscreteTimeStochasticLinearQuadraticRegulator_FiniteHorizon(Ad, Gd, Q, R, Wd, x0, Qf, P0, dt=dt, tf=10)
    
    dlqr.run_sim()
    dlqr_stoch.run_sim()
    #dkf.run_sim()

    dlqr.plot_control_gains()
    dlqr_stoch.plot_control_gains()
    dlqr.plot_ricatti()
    foo = 9

def run_dlqr_plots(dlqrs):
    lgnds = []
    plt.close()
    plt.figure()
    for dlqr in dlqrs:
        lgnd = dlqr.plot_ricatti()
        lgnds.append(lgnd)
        print("My M is: ", dlqr.m)
    plt.legend(tuple(lgnds))
    plt.title("DLQ w/ Param Var Ricatti Energy vs Time")
    plt.ylabel("K(t)")
    plt.xlabel("Time")
    plt.show()
    plt.close()
    plt.figure()
    lgnds = []
    for dlqr in dlqrs:
        lgnd = dlqr.plot_covariance()
        lgnds.append(lgnd)
    plt.legend(tuple(lgnds))
    plt.title("DLQ w/ Param Var Covariance vs Time")
    plt.ylabel("K(t)")
    plt.xlabel("Time")
    plt.show()
    plt.close()
    plt.figure()
    for dlqr in dlqrs:
        lgnd = dlqr.plot_ricatti_with_m()
        lgnds.append(lgnd)
    plt.legend(tuple(lgnds))
    plt.title("DLQ w/ Param Var Ricatti Energy using ""m"" vs Time")
    plt.ylabel("Var")
    plt.xlabel("Time")
    plt.show()
    plt.close()
    plt.figure()
    for dlqr in dlqrs:
        lgnd = dlqr.run_sim()
        lgnds.append(lgnd)
    plt.legend(tuple(lgnds))
    plt.ylabel("Reponse")
    plt.xlabel("Time")
    plt.title("DLQ w/ Param Var Response vs Time")
    plt.show()
    plt.close()

def test_uncertainty_threshold_principal():
    A = np.matrix([1.1], dtype = np.float)
    B = np.matrix([1.0], dtype = np.float)
    Q = np.matrix([1.0], dtype=np.float)
    R = np.matrix([1.0], dtype = np.float)
    Qf = np.matrix([0], dtype=np.float)
    x0 = np.matrix([1.0], dtype = np.float)
    P0 = np.matrix([.25], dtype=np.float)
    
    # Case 1: Saa and Sab are zero
    Saa = np.matrix([0]) 
    Sab = np.matrix([0])
    Sbbs = [0,.81,1.44,2.25,2.89,3.61,4.41,4.84,5.76]
    dlqrs = [DiscreteTimeStochasticLinearQuadraticRegulator_ParamVar_FiniteHorizon(A, B, Q, R, x0,P0, Qf,Saa,np.matrix([Sbbs[i]]),Sab, dt=1, tf=50)
                for i in range(len(Sbbs))]
    #run_dlqr_plots(dlqrs)
    
    # Case 2: Sab is zero
    Sbb = np.matrix([0])
    Sab = np.matrix([0])
    Saas = [0, .25, .49, .64, .81, 1.00]#, 1.21]
    dlqrs = [DiscreteTimeStochasticLinearQuadraticRegulator_ParamVar_FiniteHorizon(A, B, Q, R, x0,P0, Qf, np.matrix([Saas[i]]),Sbb, Sab, dt=1, tf=50)
             for i in range(len(Saas))]
    run_dlqr_plots(dlqrs)

    # Case 3: Sab is zero
    Sbb = np.matrix([.64])
    Sab = np.matrix([0])
    Saas = [0, .16, .25, .36, .49]#,.64, .81]
    dlqrs = [DiscreteTimeStochasticLinearQuadraticRegulator_ParamVar_FiniteHorizon(A, B, Q, R, x0, P0, Qf, np.matrix([Saas[i]]), Sbb, Sab, dt=1, tf=50)
             for i in range(len(Saas))]
    run_dlqr_plots(dlqrs)
    foo = 8

def test_CMU_Mich_Car_Control():
    
    b = .4  # Ns/m
    m = 1  # kg
    gain = .1  # throttle model

    Ac = np.matrix([[0, 1], [0, -b/m]])
    Bc = np.matrix([[0], [gain/m]])
    #Cc = np.matrix([0,1])
    Cc = np.matrix([[1, 0],[0,1]])
    Dc = np.matrix([[0],[0]])
    #Dc = np.matrix([0])
    Umag = 20
    ssc = sig.StateSpace(Ac, Bc, Cc, Dc)
    dt = .1
    ssd = sig.cont2discrete((ssc.A, ssc.B, ssc.C, ssc.D), dt)
    Ad = ssd[0]
    Bd = ssd[1]
    Cd = ssd[2]
    Dd = ssd[3]
    
    """
    # LQR Work
    Q_lqr = np.matrix([[0,0],[0,1]])
    R_lqr = np.matrix([.01])
    K_lqr, P_lqr = dlqr(Ad,Bd,Q_lqr,R_lqr)
    x0 = np.matrix([0, 0]).T
    xr = np.matrix([0, 4]).T
    t_final = 10
    sim_dlqr(Ad, Bd, Cd, Q_lqr, R_lqr, x0, xr, dt, t_final = t_final)
    """

    spectral_density = .1  # White noise
    Qc = np.matrix([spectral_density])
    F = ssd[0]
    Qd = F*np.matrix([[0, 0], [0, .01*spectral_density]])*F.T
    Vc = np.matrix([[.05, 0], [0, .025]])
    V = Vc
    #Vc = np.matrix([.05])
    x0 = np.matrix([2.5, 0]).T
    x0_truth = np.matrix([2.5, .2]).T
    P0 = np.matrix([[0.01,0],[0,.05]])
    
    Us = np.concatenate((Umag*np.ones(40), np.zeros(40), Umag*np.ones(40)))
    Us2 = [Umag*np.sin(5*t) for t in np.arange(0,10,dt)]
    kf = DiscreteKalmanFilter(Ad,Bd,Cd, dt, Qd, V, x0, P0, x0_truth)
    ckf = ContinousKalmanFilter(Ac,Bc,Cc,Qc,Vc,x0,P0,x0_truth)
    
    R = np.matrix([.01])
    Q = Cc.T*Cc
    Qf = 2*Q
    lqr = DiscreteTimeLinearQuadraticRegulator_FiniteHorizon(Ad,Bd,Q,R,x0,Qf,dt,tf = 5)
    #clqr = ContinousTimeLinearQuadraticRegulator_FiniteHorizon(Ac,Bc,Q,R,x0,Qf,tf=5)
    
    #kf.run_sim(Us2, plot = True)
    ckf.run_sim(Us2, plot = True)
    lqr.run_sim()
    clqr.run_sim()
    lqr.plot_control_gains()
    lqr.plot_ricatti()
    foo = 9

def test_CMU_Mich_Car_Control_Reference_Tracking():
    
    b = .4  # Ns/m
    m = 1  # kg
    gain = .1  # throttle model
    Umag = 5
    dt = .05
    
    # Discrete Error System Home Brew
    Ad = np.matrix([[1, dt,0,0], 
                    [0, .985,0,0],
                    [0,-1,0,1],
                    [0,0,0,1]])
    Bd = np.matrix([.5*dt**2, dt,0,0]).T
    Cd = np.matrix([[1,0,0,0],[0,1,0,0]])
    Dd = np.matrix([0, 0]).T
    x0e = np.matrix([0,0,0,8]).T

    # Augmented Error System -- Speyer:
    Aaug = np.matrix([[0, 1, 0, 0, 0],
                      [0, -b/m, 0, 0, 0],
                    [0,0,0, 1,0], 
                    [0,0,0, 0,1],
                    [0,0,0,0,-b/m]])
    Baug = np.matrix([0,gain/m,0, 0, -gain/m]).T
    Caug = np.matrix([[1,0,0,0,0],[0,1,0,0,0]])
    Daug = np.matrix([0, 0]).T
    x0aug = np.matrix([0,0,6, 3, 0]).T
    
    # Kalman Filter
    spectral_density = .15  # White noise
    Qc = np.matrix([spectral_density])
    Qd = Ad*np.matrix([[0, 0,0,0],
                      [0, .01*spectral_density,0,0],
                      [0, 0,0,0],
                      [0, 0,0,0]])*Ad.T
    Vc = np.matrix([[.5, 0], [0, .25]])
    V = Vc
    P0d = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    P0c = np.matrix([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    x0c = np.matrix([.5, 0,5,2.5,0]).T
    x0d = np.matrix([.5, 0, 0, 5]).T
    x0c_truth = np.matrix([.5, .25,5,2.5,.25]).T
    x0d_truth = np.matrix([.5, .25,0,5]).T

    kf = DiscreteKalmanFilter(Ad,Bd,Cd, dt, Qd, V, x0d, P0d, x0d_truth)
    ckf = ContinousKalmanFilter(Aaug,Baug,Caug,Qc,Vc,x0c,P0c,x0c_truth)
    Us = [Umag*np.sin(t) for t in np.arange(0,10,dt)]
    #kf.run_sim(Us, plot = True)
    #ckf.run_sim(Us, plot = True)

    # Linear Quadratic Regulator
    tf = 4
    R = np.matrix([2.5])
    Raug = np.matrix([.1])
    xref_gain = 3
    xref_tf_gain = 200
    vref_gain = 10
    vref_tf_gain = 50
    Q = vref_gain*np.matrix([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 0]])
    Qf = vref_tf_gain*Q

    Qaug = 4*np.matrix([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]])
    Qfaug = .5*Qaug
    
    dlqr = DiscreteTimeLinearQuadraticRegulator_FiniteHorizon(Ad,Bd,Q,R,x0e,Qf,dt,tf = tf)
    clqr = ContinousTimeLinearQuadraticRegulator_FiniteHorizon(Aaug,Baug,Qaug,Raug,x0aug,Qfaug,tf=tf)

    dlqr.run_sim()
    clqr.run_sim()
    foo = 9

def test_CMU_Mich_Car_Control_LQG_Reference_Tracking():
    b = .4  # Ns/m
    m = 1  # kg
    gain = .1  # throttle model
    Umag = 5
    dt = .05
    
    # Discrete Error System -- Home Brew
    Ad = np.matrix([[1, dt,0,0], 
                    [0, .985,0,0],
                    [0,-1,0,1],
                    [0,0,0,1]])
    Bd = np.matrix([.5*dt**2, dt,0,0]).T
    Cd = np.matrix([[1,0,0,0],[0,1,0,0]])
    Dd = np.matrix([0, 0]).T

    # Augmented Error System -- Speyers:
    Aaug = np.matrix([[0, 1, 0, 0, 0],
                      [0, -b/m, 0, 0, 0],
                    [0,0,0, 1,0], 
                    [0,0,0, 0,1],
                    [0,0,0,0,-b/m]])
    Baug = np.matrix([0,gain/m,0, 0, -gain/m]).T
    Caug = np.matrix([[1,0,0,0,0],[0,1,0,0,0]])
    Daug = np.matrix([0, 0]).T
    
    # Kalman Filter
    spectral_density = .15  # White noise
    Qc = np.matrix([spectral_density])
    Qd = Ad*np.matrix([[0, 0,0,0],
                      [0, .01*spectral_density,0,0],
                      [0, 0,0,0],
                      [0, 0,0,0]])*Ad.T
    Vc = np.matrix([[.5, 0], [0, .25]])
    Vd = Vc
    P0d = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    P0c = np.matrix([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

    # Initial States
    x0c = np.matrix([.2,0,6, 3, 0]).T
    x0c_truth = np.matrix([.2, -.05, 6, 3, -.05]).T
    
    x0d = np.matrix([0, .1, 0, 8]).T
    x0d_truth = np.matrix([.25, 0, 0, 8]).T

    # Linear Quadratic Regulator
    tf = 4
    R = np.matrix([2.5])
    Raug = np.matrix([.1])
    xref_gain = 3
    xref_tf_gain = 200
    vref_gain = 10
    vref_tf_gain = 50
    Q = vref_gain*np.matrix([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 0]])
    Qf = vref_tf_gain*Q

    Qaug = 4*np.matrix([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]])
    Qfaug = .5*Qaug
    

    kf = DiscreteKalmanFilter(Ad,Bd,Cd, dt, Qd, Vd, x0d, P0d)
    dlqr = DiscreteTimeLinearQuadraticRegulator_FiniteHorizon(Ad,Bd,Q,R,x0d,Qf,dt,tf = tf)

    dlqg = DiscreteTimeLinearQuadraticGaussian(Ad,Bd,Q,Qf,R,Cd,Vd,Qd,P0d,x0d,x0d_truth,dt)
    dlqg.track_ref(xref_val = 4,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 8,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 12,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 8,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 4,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 0,duration = 3, plot = False)
    dlqg.track_ref(xref_val = -4,duration = 3, plot = False)
    dlqg.track_ref(xref_val = -8,duration = 3, plot = False)
    dlqg.track_ref(xref_val = -12,duration = 3, plot = False)
    dlqg.track_ref(xref_val = -8,duration = 3, plot = False)
    dlqg.track_ref(xref_val = -4,duration = 3, plot = False)
    dlqg.track_ref(xref_val = 0,duration = 3, plot = False)
    dlqg.plot_trajectory()
    foo = 9

def test_filter_py_kalman():
    dt = .05
    # time step
    F = np.matrix([[1, dt, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, dt],
                        [0, 0, 0, 1]])
    
    G = np.matrix([dt**2/2, dt, dt**2/2, dt]).T

    H = np.matrix([[1/0.3048, 0, 0, 0],
                        [0, 0, 1/0.3048, 0]])
    
    spectral_density = 1  # White noise, for cont time spectral density
    Qc = np.matrix([[0, 0, 0, 0], 
                    [0, spectral_density, 0, 0],
                    [0,0,0,0],
                    [0,0,0,spectral_density]])
    Qd = F*Qc*F.T
    tst = Q_discrete_white_noise(4, dt=dt, var=1.)
    V = np.matrix([[.05, 0], [0, .05]])
    x0 = np.matrix([.25, .1, -.25, -.1]).T
    x0_truth = np.matrix([0, 0,0,0]).T
    P0 = np.matrix(np.eye(F.shape[0]))*5
    kf = DiscreteKalmanFilter(F,G,H,dt,Qd,V,x0,P0,x0_truth)
    Tf = 10
    times = np.arange(0,Tf,dt)
    U = [1.2*np.sin(2*t) for t in times]

    kf.run_sim(U, plot = True)
    foo = 9

def numeric(f,xsym,xvals):
    tmp = f
    for i in range(len(xvals)):
        tmp = tmp.subs([(xsym[i],xvals[i])])
    return tmp

def move2(x, u, dt, isSym = False):
    if not isSym: # Give numeric non linear function
        x = np.array(x).reshape(-1)
        u = np.array(u).reshape(-1)
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt
        wheelbase = .2
        if abs(steering_angle) >= 0.0001:  # is robot turning?
            beta = (dist / wheelbase) * np.tan(steering_angle)
            r = wheelbase / np.tan(steering_angle)  # radius
            x_k1 = np.array([[x[0]-r*np.sin(hdg) + r*np.sin(hdg + beta)],
                        [x[1] + r*np.cos(hdg) - r*np.cos(hdg + beta)],
                        [x[2]+ beta]])
            return x_k1
        else:  # moving in straight line
            dx = np.array([[dist*np.cos(hdg)],
                        [dist*np.sin(hdg)],
                        [0]])
            return x + dx
    else: # Give symbolic function representation
        hdg = x[2, 0]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt
        wheelbase = .5
        beta = (dist / wheelbase) * sym.tan(steering_angle)
        r = wheelbase / sym.tan(steering_angle)  # radius
        x_k1 = sym.Matrix([[x[0] - r*sym.sin(hdg) + r*sym.sin(hdg + beta)],
                            [x[1] + r*sym.cos(hdg) - r*sym.cos(hdg + beta)],
                            [x[2] + beta]])
        return x_k1

def move3(x, u, dt, isSym=False):
    if not isSym:  # Give numeric non linear function
        x = np.array(x).reshape(-1)
        u = np.array(u).reshape(-1)
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt
        wheelbase = .2
        if abs(steering_angle) >= 0.0001:  # is robot turning?
            beta = (dist / wheelbase) * np.tan(steering_angle)
            r = wheelbase / np.tan(steering_angle)  # radius
            x_k1 = np.array([[x[0]-r*np.sin(hdg) + r*np.sin(hdg + beta)],
                             [x[1] + r*np.cos(hdg) - r*np.cos(hdg + beta)],
                             [x[2] + beta],
                             [-x[0] + x[4]],
                             [x[4]],
                             [-x[1] + x[6]],
                             [x[6]],
                             [-x[2] + x[8]],
                             [x[8]]])
            
            return x_k1
        else:  # moving in straight line
            dx = np.array([[x[0] + dist*np.cos(hdg)],
                           [x[1] + dist*np.sin(hdg)],
                           [0],
                           [-x[0] + x[4]],
                           [x[4]],
                           [-x[1] + x[6]],
                           [x[6]],
                           [-x[2] + x[8]],
                           [x[8]]])
            return dx
    else:  # Give symbolic function representation
        hdg = x[2, 0]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt
        wheelbase = .2
        beta = (dist / wheelbase) * sym.tan(steering_angle)
        r = wheelbase / sym.tan(steering_angle)  # radius
        x_k1 = sym.Matrix([[x[0] - r*sym.sin(hdg) + r*sym.sin(hdg + beta)],
                           [x[1] + r*sym.cos(hdg) - r*sym.cos(hdg + beta)],
                           [x[2] + beta],
                           [-x[0] + x[4]],
                           [x[4]],
                           [-x[1] + x[6]],
                           [x[6]],
                           [-x[3] + x[8]],
                           [x[8]]])
        return x_k1

# Assumes x y theta first three states
def measure_move2(x,dt, isSym = False):
    landmark_pos = (5,5)
    x = np.array(x).reshape(-1)
    if not isSym:
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
        hx = np.matrix([[dist],
                    [np.arctan2(py - x[1], px - x[0]) - x[2]] ])
        return hx
    else:
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = sym.sqrt((px - x[0])**2 + (py - x[1])**2)
        hx = sym.Matrix([[dist],
                    [sym.atan2(py - x[1], px - x[0]) - x[2]]])
        return hx

def test_filter_py_ekf():
    x,y,theta = sym.symbols('x,y,theta')
    state = sym.Matrix([x,y,theta])
    vel_inp, steer_inp = sym.symbols('u, steer_ang')
    input_u = sym.Matrix([vel_inp,steer_inp])
    xu_sym = state.col_join(input_u)
    dt = .2
    x0 = np.matrix([0, 0, 0]).T
    x0_truth = np.matrix([.1,.1,0]).T
    u0 = np.matrix([0,0]).T
    spectral_density = .00001
    Qc = np.matrix([[spectral_density, 0, 0],
                    [0, spectral_density, 0],
                    [0, 0, 0]])
    #Qd = np.matrix([[spectral_density, 0],
    #                [0, .1e-3*spectral_density]])

    Fsym = move2(state,input_u, dt, isSym=True).jacobian(state)
    Fnum = np.matrix(numeric(Fsym,xu_sym,np.concatenate((x0,u0))), dtype = np.float)
    Qd = Fnum*Qc*Fnum.T
    V = np.matrix([[.001, .000056], [.000056, 1e-3]])
    P0 = np.matrix([[1,0,0],[0,1,0],[0,0,0]])
    ekf = DiscreteExtendedKalmanFilter(x_sym=state,u_sym=input_u,f_xu=move2,h_x=measure_move2,
                                       dt=dt, Q=Qd, V=V, x0=x0, P0=P0, x0_truth=x0_truth, lininearize_step=5)
    Tf = 8
    times = np.arange(0, Tf, dt)
    U1 = [np.array([np.asscalar(.5+ 5*np.sin(.5*t)),np.asscalar(.0034 + .0003*t)],dtype = np.float) for t in times]
    t = time.time()
    ekf.run_sim(U1, plot=True)
    print(time.time() -t)
    foo = 9

def test_CMU_Mich_Car_Control_Extended_LQG_Reference_Tracking():
    # Symbolic Dynamic Variables
    x, y, theta,xerr,xref,yerr,yref,theta_err,theta_ref = sym.symbols('x,y,theta,xerr,xref,yerr,yref,theta_err,theta_ref')
    state = sym.Matrix([x, y, theta,xerr,xref,yerr,yref,theta_err,theta_ref])
    vel_inp, steer_inp = sym.symbols('u, steer_ang')
    input_u = sym.Matrix([vel_inp, steer_inp])
    xu_sym = state.col_join(input_u)
    
    # Kalman Setup -- State and Error dynamics with reference added into state
    dt = .05
    spectral_density = 1e-12
    weights = np.matrix([1,1,0,0,0,0,0,0,0])
    weights2 = np.matrix([1,1,0,0,0,0,0,0,0])
    Qd = np.matrix(np.diag(weights))*spectral_density
    V = np.matrix([[1e-6, 0], [0, 1e-7]])
    P0 = .1*np.matrix(np.eye(9))*weights2.T*weights2
    
    # Setup Tracker -- Weight Error Dynamics
    weights3 = np.array([0,0,0,1,0,1,0,1,0])
    Q = np.matrix(np.diag(weights3))
    Qf = 2*Q
    R = np.matrix([[.01,0],[0,2]])
    
    xr = 1
    yr = 1
    thetar = np.arctan2(yr, xr)
    xrefs = [xr,yr,np.arctan2(yr,xr)] # x, y, theta
    x0 = np.matrix([.1, .1, .1, xrefs[0],
                    xrefs[0], xrefs[1], xrefs[1], xrefs[2], xrefs[2]]).T
    x0_truth = np.matrix([.1 + .01, .1 + .01, .1, xrefs[0],xrefs[0],xrefs[1],xrefs[1],xrefs[2],xrefs[2]]).T
    tf = .5
    delqg = DiscreteTimeExtendedLinearQuadraticGaussian(f_xu=move3,h_x=measure_move2,Q=Q,
                                                        R=R,Qf=Qf,x_sym=state,u_sym=input_u,
                                                        Q_noise=Qd,V = V,x0=x0,dt = dt,
                                                        P0 = P0,x0_truth = x0_truth, linearize_step= 1)
    delqg.track_references(xrefs,tf,plot = True)
    delqg.dlqr.plot_control_gains()
    delqg.dlqr.plot_ricatti()
    foo = 9

    #ekf = DiscreteExtendedKalmanFilter(x_sym=state, u_sym=input_u, f_xu=move3, h_x=measure_move2,
    #                                   dt=dt, Q=Qd, V=V, x0=x0, P0=P0, x0_truth=x0_truth, lininearize_step=1)

    #times = np.arange(0, Tf, dt)
    #U1 = [np.array([np.asscalar(.5 + 5*np.sin(.5*t)),
    #                np.asscalar(.0034 + .0003*t)], dtype=np.float) for t in times]
    #ekf.run_sim(U1, plot=True)



# The difference in amplitude for discrete and regular impulse is 10  
if __name__ == "__main__":
    #test_kalman_ukf()
    #test_filter_py_kalman()
    #test_lq_stochastic_ctime()
    #test_uncertainty_threshold_principal()
    #test_CMU_Mich_Car_Control()
    #test_CMU_Mich_Car_Control_Reference_Tracking()
    #test_CMU_Mich_Car_Control_LQG_Reference_Tracking()
    #test_filter_py_ekf()
    test_CMU_Mich_Car_Control_Extended_LQG_Reference_Tracking()
    foo = 9
    
    
    












