# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:33:24 2018

@author: natsn
"""

import numpy as np
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.image as im


#Load in the image with matplotlibs image processing tools
img = im.imread('lena.png')
bw = img.mean(axis = 2)


Hx = np.atleast_2d(np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = np.float32))

Hy = Hx.T

Gx = convolve2d(bw,Hx)
Gy = convolve2d(bw,Hy)

G = np.sqrt(Gx*Gx + Gy*Gy)

plt.imshow(G, cmap = 'gray')
plt.show()

# Now plot the gradients direction
theta = np.arctan2(Gy,Gx)

plt.imshow(theta, cmap = 'gray')
plt.show()


