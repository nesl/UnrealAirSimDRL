# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:53:24 2018

@author: natsn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from datetime import datetime

#Convolutional Algorithm
# Convolution works between two arguments. In each direction the array is defined

img = im.imread('Lena.png')

# Convert the RGB image into a grayscale image
bw_im = img.mean(axis = 2)

# Define the convolutional filter that you'd like to use
W = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        W[i,j] = (i - 4.5)**2 + (j - 4.5)**2

X = np.array([3,4,5])
w = np.array([2,1])
g = convolve1d(X,w)
# Begin 1D convolution
def convolve1d(X,w):
    n1 = np.size(X)
    m1 = np.size(w)
    # g[t] = h[t-s]x[s]
    g = np.zeros(n1 + m1 -1)
    for i in range(n1 + m1 -1): # slide the filter over the input vector
        for j in range(m1):
            if(not (i - j) < 0) and (i-j) < m1 and j < n1:
                g[i] += w[i-j]*X[j] # slide the kernal across the input vector
    return g


def convolve2d(X,w,mode = 'full'):
    len_xi, len_xj = np.shape(X) # returns a tuple of the shape
    len_wi, len_wj = np.shape(w)
    
    # Start Time
    t0 = datetime.now()
    #g[ti,tj] = h[ti-s1, tj-s2]*x[s1,s2]
    # Horizontal kernal shift across with a sum on each
    # vertical kernal shift down after each horizontal kernal shift
    # Increment starting position of kernal for x axis 
    # Increment starting position of kernal for y axis
    
    offset = (0,0)
    if mode == 'full': 
        offset = (0,0)
    elif mode == 'same': # Not working
        offset = (len_wi - 1, len_wj -1)
    else:
        print("Error in mode parameter setting")
    
    # declare convolutional output g 
    g = np.zeros((len_xi+len_wi-1 -offset[0],len_xj+len_wj-1 - offset[1]))
    for ti in range(len_xi + len_wi - 1):
        for tj in range(len_xj + len_wj - 1):
                    # Assert Boudaries:
                    # (ti - si) must stay postive
                    # (tj - sj) must stay postive
                    # s1 cant go over the value of len_wi 
                    # s2 cant go over the value of len)_wj
                    if( not (ti + offset[0]) < 0 and not (tj + offset[1]) < 0
                    and (ti + offset[0]) < len_xi and (tj + offset[1]) < len_xj):
                        g[ti:ti+len_wi,tj:tj+len_wi] += X[ti + offset[0], tj + offset[1]]*w
    t1 = datetime.now()
    tm = t1 - t0
    print("Elaplsed Time is: ", tm)
    return g
                        
    
    
def autocorrelation(tau):
    pass

def crosscorrelatiom(tau):
    pass















    