# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:43:13 2017

@author: natsn
"""
import numpy as np
import pandas as pd
import scipy as sci
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D
# Generating all project data in single class so any updates can be done in a single place
from sympy import Matrix, cos,sin, symbols
import control


# returns two lists, x and y values
def unit_circle():
    x = []
    y = []
    w_max = 2*np.pi
    ws = np.arange(0, w_max, .05)
    for w in ws:
        x.append(np.cos(w))
        y.append(np.sin(w))
    return x, y

# Tuple of arrays


def stack_arrays(us):
    assert type(us) == tuple
    uv = np.vstack(us)
    uv = np.matrix(uv)
    uvs = [uv[:, i] for i in range(uv.shape[1])]
    return uvs


def preprocess_pulse_repsonse_data(us, ys):
    #remove any offsets in output data using data prior to pulse application
    num_channels = int(len(ys) / len(us))
    _us = None
    _ys = None
    for i in range(len(us)):
        u = np.array(us[i], dtype=np.float)
        tpulse = np.argmax(u)
        u_max = np.max(u)
        u /= u_max
        for j in range(num_channels):
            y_pp_mean = np.mean(ys[num_channels*i + j][0:tpulse-1])
            # Subtract mean perturbation before pulse happens from y
            ys[num_channels*i + j] = ys[num_channels*i + j] - y_pp_mean
            # Rescale IO data so pulse has magnitude 1
            ys[num_channels*i + j] / np.max(us[i])
    return us, ys


def get_input_output_data(fn='ProjectData.csv'):
    data = pd.read_csv('ProjectData.csv', delimiter=',')  # comma seperated
    u1 = data['u1'].values
    u2 = data['u2'].values
    y11 = data['y11'].values
    y21 = data['y21'].values
    y12 = data['y12'].values
    y22 = data['y22'].values
    return [u1, u2], [y11, y21, y12, y22]


def plot_SISO(t, y, title="Y vs T"):
    plt.plot(t, y)
    plt.title(title)
    plt.show()


def plot_MIMO(us, ys, dt):
    num_figs = len(us)
    num_ops = len(ys)
    num_channels = int(num_ops / num_figs)
    #Plot Input Response
    plt.figure(1)
    for j in range(num_figs):
        time = np.arange(0, dt*len(us[j]), dt)
        plt.subplot(num_channels, 1, j+1)
        plt.scatter(time, us[j])
        plt.title("Input Signal u"+str(j+1))
        plt.xlabel("time")
        plt.ylabel("u"+str(j))

    # Plot Output Responses
    for i in range(num_figs):
        plt.figure(i+2)
        for j in range(num_channels):
            time = np.arange(0, dt*len(ys[num_channels*i + j]), dt)
            plt.subplot(num_channels, i+1, j+1)
            plt.scatter(time, ys[num_channels*i + j])
            plt.title("Response y"+str(i+1)+str(j+1))
            plt.xlabel("u"+str(i+1))
            plt.ylabel("y"+str(i)+str(j))
    # 3D View Of input output and time
    # Add Legend
    for i in range(num_figs):
        fig = plt.figure(i + 2 + num_figs)
        ax = Axes3D(fig)
        for j in range(num_channels):
            time = np.arange(0, dt*len(ys[num_channels*i + j]), dt)
            ax.scatter3D(time, us[i], ys[num_channels*i + j])
            ax.set_title("Time, " + "Response y to Input Signal u"+str(i+1))
            ax.set_xlabel("u"+str(i+1))
            ax.set_ylabel("y")
    plt.show()

# place impulse responses in ascending order, per channel


def form_hankel_matrix(ys, indices=None, m=2, q=2):
    #Assert all input observations are same length
    assert len(ys) == m*q
    assert type(ys) == list
    for i in range(len(ys)-1):
        if len(ys[i]) != len(ys[i+1]):
            print("All of your observations must be same length!")
            break
    if indices is None:
        indices = []
        indices.append(0)
        indices.append(len(ys[0]))

    datachunk = [i for i in range(indices[0], indices[1])]
    hchunk = np.array(np.zeros((len(datachunk)*m, q)))  # initially
    for col in range(0, len(datachunk)):
        htemp = np.array(np.zeros((m, q)))
        for t in datachunk:
            hkls = []
            for y in ys:
                hkls.append(y[t+col])
            if q > 1:
                hk = np.array(hkls).reshape((m, q)).T
            else:
                hk = np.array(hkls).reshape((m, q))
            # initialize chunk
            if t == datachunk[0]:
                htemp = hk
            # else concatenate to chunk
            else:
                htemp = np.concatenate((htemp, hk), axis=0)
        if col == 0:
            hchunk = htemp
        else:
            hchunk = np.concatenate((hchunk, htemp), axis=1)

    return np.matrix(hchunk)  # return the hankel matrix


# Ho of the hankel matrix is when the pulse occurs, so remember to have first indice has pulse stop + 1
def system_identification(ys, indices, m, q, state_factorizations=[2, 4, 6, 8]):
    # 1: Take the pulse repsonse data and form hankel
    # 2: Create SVD of Hankel
    #**User selects all the possible factorizations they want
    #**For each 'ns' state factorization..
    # 3: Choose a factorization for observability and controlability matrices
    # 4: Make another hankel (hankel of k+1) -> H_tilde
    # 6: B and C are found by the first ns
    # 6: A left and right inverse of H_tilde with the found obs and controllable matrices recovers A
    hankel = form_hankel_matrix(ys, indices=indices, m=2, q=2)
    hankel_tilde = form_hankel_matrix(
        ys, indices=[indices[0]+1, indices[1]+1], m=m, q=q)
    U, S, Vh = np.linalg.svd(hankel)
    U = np.matrix(U)
    #S = np.matrix(S)
    Vh = np.matrix(Vh)
    state_models = {}
    state_models["Hankel"] = hankel
    state_models["Hankel_tilde"] = hankel_tilde
    for sf in state_factorizations:
        # SVD Hankel System Identification
        state_models["ns"+str(sf)] = {}
        state_models["ns"+str(sf)]["Sing_Vals"] = S[0:sf]
        state_models["ns"+str(sf)]["ObsvMat_ns"] = U[:, 0:sf] * \
            np.matrix(np.sqrt(np.diag(S[0:sf])))
        state_models["ns"+str(sf)]["CtrbMat_ns"] = np.matrix(
            np.sqrt(np.diag(S[0:sf])))*Vh[0:sf, :]
        state_models["ns"+str(sf)]["B_ns"] = state_models["ns" +
                                                          str(sf)]["CtrbMat_ns"][:, 0:q]
        state_models["ns"+str(sf)]["C_ns"] = state_models["ns" +
                                                          str(sf)]["ObsvMat_ns"][0:m, :]
        state_models["ns"+str(sf)]["A_ns"] = state_models["ns"+str(sf)]["ObsvMat_ns"].getI(
        )*hankel_tilde*state_models["ns"+str(sf)]["CtrbMat_ns"].getI()

        # Grammians of the pulse response
        state_models["ns"+str(sf)]["ObsvGram_ns"] = state_models["ns" +
                                                                 str(sf)]["ObsvMat_ns"].T*state_models["ns"+str(sf)]["ObsvMat_ns"]
        state_models["ns"+str(sf)]["CtrbGram_ns"] = state_models["ns" +
                                                                 str(sf)]["CtrbMat_ns"].T*state_models["ns"+str(sf)]["CtrbMat_ns"]

    svs = S[0:state_factorizations[-1]]
    plt.scatter([i for i in range(len(svs))], svs)
    plt.show()
    return state_models


def auto_correlation(u, k_interval):
    Rxx = []
    temp = 0
    q = 0
    len_u = len(u)
    stop = 0
    start = 0
    set_temp = True
    for k in k_interval:
        q = 0
        set_temp = True
        if k < q:
            start = np.abs(k)
            stop = len(u)
        elif k > q:
            start = 0
            stop = len(u) - k
        else:
            start = 0
            stop = len(u)
        for q in range(start, stop):
            if set_temp:
                temp = u[k+q]*u[q].transpose()
                set_temp = False
            else:
                temp += u[k+q]*u[q].transpose()
        temp /= len_u
        Rxx.append(temp)
    return Rxx

# input u is a temporal list of matrices
# input y is a temporal list of observed states


def cross_correlation(u, y, k_interval):
    Rxx = []
    temp = 0
    q = 0
    len_u = len(u)
    stop = 0
    start = 0
    set_temp = True
    for k in k_interval:
        q = 0
        set_temp = True
        if k < q:
            start = np.abs(k)
            stop = len(u)
        elif k > q:
            start = 0
            stop = len(u) - k
        else:
            start = 0
            stop = len(u)
        for q in range(start, stop):
            if set_temp:
                temp = y[k+q]*u[q].transpose()
                set_temp = False
            else:
                temp += y[k+q]*u[q].transpose()
        temp /= len_u
        Rxx.append(temp)
    return Rxx


def dlqr_infinite_horizon(A, B, Q, R):
    P = la.solve_discrete_are(A, B, Q, R)
    K = R.I*B.T*P  # Optimal Control
    return K, P


def lqr_infinite_horizon(A, B, Q, R):
    P, _ = la.solve_continuous_are(A, B, Q, R)
    K = R.I*B.T*P
    return K, P

def dlqr_finite_horizon(A,B,Q,R,Qf):
    pass

# Infinite Horizon
def sim_dlqr(A, B, C, Q, R, x0, xr, dt, t_final):
    K, _ = dlqr_infinite_horizon(A, B, Q, R)
    times = np.arange(0, t_final, dt)
    xc = x0
    xs = []
    ys = []
    us = []
    for _ in times:
        u = K*(xr - xc)
        xk_1 = A*xc + B*u
        yk = C*xk_1
        ys.append(yk)
        xs.append(xk_1)
        us.append(u)
        xc = xk_1
    # Plot Response
    for i in range(ys[0].shape[0]):
        plt.plot(times, [np.asscalar(y[i, 0]) for y in ys])
        plt.show()
    return (xs, ys, us)

# Create PID Response


def discrete_pid_response(sysd, x0, xr, times, kp, ki=None, kd=None):
    zero_vector = np.matrix(np.zeros(len(x0))).T
    A = sysd[0]
    B = sysd[1]
    C = sysd[2]
    if ki is None:
        ki = 0
    if kd is None:
        kd = 0
    xc = x0  # current
    xl = x0  # last
    edl = zero_vector
    dt = times[1] - times[0]
    ei = 0
    xs = []
    ys = []
    us = []
    #xs.append(x0)
    for t in times:
        ep = kp*C*(xr - xc)
        ed = kd*C*((xr-xc) - edl) / dt
        ei += ki*C*(xr-xc) * dt

        u = ep+ed+ei
        xk_1 = A*xc + B*u
        yk = C*xk_1
        xc = xk_1
        edl = xr - xc
        xs.append(xc)
        ys.append(yk)
        us.append(u)
    for i in range(ys[0].shape[0]):
        plt.plot(times, [np.asscalar(y[i, 0]) for y in ys])
        plt.show()
    return (xs, ys, us)


def continous_time_impulse_response(A,B,C,D = None, plot = True):
    dt = .025
    T = 10
    times = np.arange(0,T,dt)

    ys = [C*la.expm(A*t)*B + D if D is not None else 0 for t in times]
    xs = [la.expm(A*t)*B + D if D is not None else 0 for t in times]
    if plot:
        for i in range(ys[0].shape[0]):
            plt.plot(times, [np.asscalar(y[i, 0]) for y in ys])
            plt.title("Response Y" + str(i))
            plt.show()
    return (xs, ys, times)

def continous_time_step_response(A,B,C,D, plot = True):
    ss = sig.StateSpace(A,B,C,D)
    t,ys  = sig.step(system=ss)

    if plot:
        for i in range(ys.shape[1]):
            plt.plot(t, ys[:,i])
            plt.title("Response Y" + str(i))
            plt.show()
    return (t, ys)

# Input is of form: B.T*e^(A(t1 - t)*Gc.I*x1)
def continous_time_least_norm_input(A,B,xr,t1,dt):
    times = np.arange(0,t1,dt)
    Gc = gram_ctrb(A,B)
    uln_t = [B.T*la.expm(A.T*(t1 - t)) * Gc.I * xr for t in times]
    return uln_t
    

def continous_time_frequency_repsonse(A, B, C, plot=True):
    w_range = np.arange(0, 150, .1)  # radians / second
    e_jwt = [1j*w for w in w_range]
    num_responses = C.shape[0]
    num_inputs = B.shape[1]
    # Create Complex Frequency Response
    freq_resp_raw = [
        C*np.matrix(e*np.matrix(np.eye(A.shape[0])) - A).getI()*B for e in e_jwt]

    # Attain magnitude and angle
    freq_resp_mag = [np.abs(fr) for fr in freq_resp_raw]
    freq_resp_angle = [np.angle(fr, deg=True) for fr in freq_resp_raw]
    print(freq_resp_mag[0:10])
    if plot:
        plt.figure()
        k = 1
        for i in range(num_responses):
            for j in range(num_inputs):
                plt.subplot(2*num_responses, num_inputs, k)
                plt.loglog(w_range, [np.asscalar(frm[j, i])
                                     for frm in freq_resp_mag], k)
                plt.title("Mag Freq Response y" + str(j+1) + str(i+1))
                plt.xlabel("wnyq_rad")
                plt.ylabel("Mag y" + str(j+1) + str(i+1))
                k += 1
                plt.subplot(2*num_responses, num_inputs, k)
                plt.semilogx(w_range, [fra[j, i]
                                       for fra in freq_resp_angle], k)
                plt.title("Phase Freq Response y" + str(j+1) + str(i+1))
                plt.xlabel("wnyq_rad")
                plt.ylabel("Phase y" + str(j) + str(i))
                k += 1
        plt.show()
    return (w_range, freq_resp_raw, freq_resp_mag, freq_resp_angle)

def continous_time_response(A,B,C,U,dt):
    pass

# returns the full x and y state information for each and every state k0 -> # rounds
# we should know the sampling time interval and how it connects with how many rounds we play
# input a forcing term u that contains the matrix forcing history for the system at each timestep


def discrete_time_response(A, B, C, U, x0, dt, plot=True):
    x_rounds = []
    x_rounds.append(x0)
    y_rounds = []
    y_rounds.append(C*x0)

    xk = x0
    for uk in U:
        xk_1 = A*xk + B*np.matrix(uk)
        x_rounds.append(xk_1)
        yk_1 = C*xk_1
        y_rounds.append(yk_1)
        xk = xk_1
    time = np.arange(0, dt*(len(U)+1), dt)
    if plot == True:
        for i in range(C.shape[0]):
            plt.subplot(C.shape[0], 1, i+1)
            plt.scatter(time, [yr[i, 0] for yr in y_rounds])
            plt.title("Response y" + str(i))
            plt.ylabel("Response")
            plt.xlabel("Time")
        plt.show()
    return x_rounds, y_rounds, time


def discrete_time_frequency_repsonse(A, B, C, wnyq_hz, dt, plot=True):
    w_max = wnyq_hz * 2*np.pi
    w_range = np.arange(0, w_max, .1)  # radians / second
    e_jwt = [np.exp(1j*w*dt) for w in w_range]
    num_responses = C.shape[0]
    num_inputs = B.shape[1]
    # Create Complex Frequency Response
    freq_resp_raw = [
        C*np.matrix(e*np.matrix(np.eye(A.shape[0])) - A).getI()*B for e in e_jwt]

    # Attain magnitude and angle
    freq_resp_mag = [np.abs(fr) for fr in freq_resp_raw]
    freq_resp_angle = [np.angle(fr, deg=True) for fr in freq_resp_raw]
    print(freq_resp_mag[0:10])
    if plot:
        plt.figure()
        k = 1
        for i in range(num_responses):
            for j in range(num_inputs):
                plt.subplot(2*num_responses, num_inputs, k)
                plt.loglog(w_range, [np.asscalar(frm[j, i])
                                     for frm in freq_resp_mag], k)
                plt.title("Mag Freq Response y" + str(j+1) + str(i+1))
                plt.xlabel("wnyq_rad")
                plt.ylabel("Mag y" + str(j+1) + str(i+1))
                k += 1
                plt.subplot(2*num_responses, num_inputs, k)
                plt.semilogx(w_range, [fra[j, i]
                                       for fra in freq_resp_angle], k)
                plt.title("Phase Freq Response y" + str(j+1) + str(i+1))
                plt.xlabel("wnyq_rad")
                plt.ylabel("Phase y" + str(j) + str(i))
                k += 1
        plt.show()
    return (w_range, freq_resp_raw, freq_resp_mag, freq_resp_angle)


def discrete_time_frequency_response_fourier(us, ys, dt, plot=True):
    k = 1
    om_yffts = []
    for i in range(len(us)):
        for j in range(int(len(ys)/len(us))):
            yfft = sci.fftpack.fft(
                ys[int(len(ys)/len(us))*i + j]) / sci.fftpack.fft(us[i])
            om = [i/(dt*len(yfft)) for i in range(len(yfft))]
            om_yffts.append((om, yfft))
            if plot:
                plt.subplot(2*int(len(ys)/len(us)), len(us), k)
                plt.loglog(om, np.abs(yfft))
                plt.title("Emperical Freq Resp Mag y" + str(j+1) + str(i+1))
                plt.xlabel("Omega rad/sec")
                plt.ylabel("Response y" + str(j+1) + str(i+1))
                k += 1
                plt.subplot(2*int(len(ys)/len(us)), len(us), k)
                plt.semilogx(om, np.angle(yfft, deg=True))
                plt.title("Emperical Freq Resp Mag y" + str(j+1) + str(i+1))
                plt.xlabel("Omega rad/sec")
                plt.ylabel("Response y" + str(j+1) + str(i+1))
                k += 1
    if plot:
        plt.show()
    return om_yffts


def pole_zeros(A, B, C, D=None, plot=True):
    assert A.shape[1] == C.shape[1]
    if D is None:
        D = np.zeros((C.shape[0], B.shape[1]))
    # attempt to solve the generalized eigenvalue problem
    temp1 = np.matrix(np.concatenate((A, B), axis=1))
    temp2 = np.matrix(np.concatenate((-1*C, -1*D), axis=1))
    S = np.matrix(np.concatenate((temp1, temp2), axis=0))
    eVals, _ = np.linalg.eig(A)

    temp1 = np.diag(np.ones((A.shape[0])))
    temp2 = np.zeros((B.shape[0], B.shape[1]))
    temp3 = np.zeros((C.shape[0], A.shape[1]+B.shape[1]))
    temp1 = np.concatenate((temp1, temp2), axis=1)
    RHS = np.matrix(np.concatenate((temp1, temp3), axis=0))

    #1) finding the eigenvalues and transmission zeros
    trans_zeros, _ = la.eig(S, RHS)  # gen eigenvalues
    trans_zeros = np.sort(trans_zeros)[0:5]

    if plot:
        plt.figure()
        xc, yc = unit_circle()
        plt.plot(xc, yc)
        plt.scatter([np.real(e) for e in eVals], [np.imag(e)
                                                  for e in eVals], c='r')
        plt.scatter([np.real(e) for e in trans_zeros], [np.imag(e)
                                                        for e in trans_zeros], c='g')
        plt.show()

    return (eVals, trans_zeros)


def discrete_to_continous_eigenvalues(evals_discrete, dt):
    # obtain real parts of trans zeros for graphing
    cont_eigs = [np.log(e)/dt for e in evals_discrete]
    return cont_eigs


def continous_to_discrete_eigenvalues(evals_cont, dt):
    return [np.exp(1j*e*dt) for e in evals_cont]


def ctrb(A, B):
    return control.obsv(A, B)

def obsv(A,C):
    return control.obsv(A,C)

def gram_obsv(A, C):
    return la.solve_continuous_lyapunov(A.T, C.T*C)

def gram_ctrb(A,B):
    return la.solve_continuous_lyapunov(A, B*B.T)




def test():
    dt = .025
    w_hz = 1/dt
    wnqy_hz = w_hz / 2
    us, ys = get_input_output_data()
    us, ys = preprocess_pulse_repsonse_data(us, ys)
    state_models = system_identification(
        ys, [41, 141], m=2, q=2, state_factorizations=[6, 7, 10, 20])
    A = state_models["ns7"]["A_ns"]
    B = state_models["ns7"]["B_ns"]
    C = state_models["ns7"]["C_ns"]
    x0 = np.matrix(np.zeros((7, 1)))
    u_zero = np.zeros(len(us[0]))
    U = stack_arrays((us[0], u_zero))
    #y_resp = discrete_time_response(A,B,C,U,x0 = x0, dt = dt, plot = True)
    #dtfr_data = discrete_time_frequency_repsonse(A,B,C, wnyq_hz = 20, dt=dt, plot = True)
    #dtfr_fourier_data = discrete_time_frequency_response_fourier(us, ys,dt, plot = True)
    #plot_MIMO(us, ys, dt)
    evals, trans_zeros = pole_zeros(A, B, C, plot=True)
    evals, trans_zeros = pole_zeros(A, B[:, 1], C[1, :], plot=True)
    cont_eigs = discrete_to_continous_eigenvalues(evals, dt)
    print("Cont Eigs: ", cont_eigs)
    k = 0
    

def cruise_control():
    m = 1000
    b = 50
    u = 500

    A = np.matrix([[0,.99],[0,-b/m]]) 
    B = np.matrix([[0],[u/m]])
    C = np.matrix(np.eye(2))
    D = np.matrix([[0],[0]])
    evals, evecs = np.linalg.eig(A)
    pole_zeros(A,B,C[1,:])
    ss = sig.StateSpace(A,B,C,D)
    continous_time_impulse_response(A,B,C,D, plot = True)
    continous_time_step_response(A,B,C,D,plot = True)
    #continous_time_frequency_repsonse(A,B,C, plot = True)
    Gc = gram_ctrb(A,B)
    U = continous_time_least_norm_input(A,B,np.matrix([0,10]),5,.025)

    #Go = gram_obsv(A,C)
    foo = 0
def InvertedPendulum():
    x0 = 0
    dx0 = 0
    theta0 = np.pi
    dtheta0 = 0
    M = .5
    m = 0.2
    b = 0.1
    I = 0.006
    g = 9.8
    l = 0.3
    F = symbols('F')
    x, theta, dx, dtheta = symbols('x, theta, dx, dtheta')
    X = Matrix([x,dx,theta,dtheta])
    U = Matrix([F])
    RHS = Matrix([[M + m,-m*l*cos(theta)],[m*l*cos(theta), I + m*l**2]])
    LHS1 = RHS.inv()*Matrix([-b*x + m*l*dtheta**2*sin(theta), -m*l*g*sin(theta)]) #Dynamics
    LHS2 = RHS.inv()*Matrix([F,0]) #Input Force
    A = LHS1.jacobian(X).subs([(x,x0),(theta,theta0),(dx,dx0),(dtheta,dtheta0)])
    B = LHS2.jacobian(U).subs([(x,x0),(theta,theta0),(dx,dx0),(dtheta,dtheta0)])
    A = np.reshape(np.matrix(np.vstack((np.matrix([0, 1, 0, 0]),
                    A[0,:],
                    np.matrix([0,0,0,1]),
        A[1, :])), dtype=np.float), (4, 4))
    B = np.matrix([0,B[0,0],0,B[1,0]], dtype = np.float).T
    C = np.matrix([[1,0,0,0],[0,0,1,0]])
    D = np.matrix([[0],[0]])
    ss = sig.StateSpace(A,B,C,D)
    dt = .025
    ssd = sig.cont2discrete((A,B,C,D),dt)
    tc, yc = sig.impulse(ss)
    td, yd = sig.dimpulse(ssd)
    plt.plot(td,[y[0,0]for y in yd])
    plt.show()
    plt.plot(td, [y[1, 0]for y in yd])
    plt.show()
    f00 = 9


def mass_spring_damper():
    m = .25
    k = 10 # n/m
    b = .2 # Nm/s

    A = np.matrix([[0,1],[-k/m, -b/m]])
    B = np.matrix([0,1/m]).T
    C = np.matrix([1,0])
    D = np.matrix([0])
    dt = .025

    ss = sig.StateSpace(A,B,C,D)
    ssd = sig.cont2discrete((A,B,C,D),dt)
    ctrb_rank = np.linalg.matrix_rank(ctrb(A,B))



def MotorControl():
    J = .01  # kgm**2
    b = .1  # N.m.s
    Ke = .01  # V/rad/sec
    Kt = .01  # N.m/Amp
    R = 1  # Electrical resistance
    L = .5  # Henries
    K = Kt  # or Ke, Ke == Kt here
    # State Space Model
    A = np.matrix([[-b/J, K/J], [-K/L, -R/L]])
    B = np.matrix([[0], [1/L]])
    C = np.matrix([1, 0])
    D = np.matrix([0])
    dt = .025  # Discrete Sample Time

    ss = sig.StateSpace(A, B, C, D)
    ssd = sig.cont2discrete((A, B, C, D), dt)
    Ad = ssd[0]
    Bd = ssd[1]
    Cd = ssd[2]
    evc, evecc = np.linalg.eig(A)
    evd, evecd = np.linalg.eig(Ad)
    # Create Bode
    #w, mag, phase = sig.bode(ss)
    #plot_SISO(w,mag, "Freq Response Mag vs w")
    #plot_SISO(w, phase, "Freq Response Phase vs w")
    #discrete_time_frequency_repsonse()
    # Transfer Function
    tf = ss.to_tf()
    print(tf)
    t, y = sig.step(ss)
    td, yd = sig.dstep(ssd)
    yd = np.reshape(yd, -1)
    #plot_SISO(t,y)
    #plot_SISO(td, yd)
    #pole_zeros(A, B, C, plot=True)
    #pole_zeros(Ad, Bd, Cd, plot=True)
    T = 10
    N = T/dt
    times = np.arange(0, T, dt)
    U = np.ones(int(N))
    x0 = np.matrix([0, 0]).T
    xr = np.matrix([2.5, 0])
    kp = 15
    kd = 4
    ki = 100
    Q = Cd.T*Cd
    R = np.matrix([.0005])
    discrete_time_response(Ad, Bd, Cd, U, x0, dt, plot=True)
    discrete_time_frequency_repsonse(Ad, Bd, Cd, 20, dt)
    continous_time_frequency_repsonse(A, B, C)
    discrete_pid_response((Ad, Bd, Cd), x0, xr, times, kp, ki, kd)
    sim_dlqr(Ad, Bd, Cd, Q, R, x0, xr, dt, T)
    x = 9


def main():
    mass_spring_damper()


if __name__ == "__main__":
    main()
