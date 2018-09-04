# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 18:25:38 2018

@author: natsn
"""
# Imports
import numpy as np

# Return a Gaussian Image Kernal
# input the size of the kernal you want (nxmxw)
def get_gaussian_kernal(sz, intensity = 1):
    if(len(sz) == 1):
        std_dev_i = np.std([i for i in range(sz[0])])
        kern = np.zeros(sz)
        for i in range(sz[0]):
            dist = (i - sz[0]/2)**2
            kern[i] = np.exp(-dist/(std_dev_i**2 * intensity) )
            energy = np.sqrt(np.sum(np.power(kern,2)))
        return kern / energy
    
    elif(len(sz) == 2):
        std_dev_i = np.std([i for i in range(sz[0])])
        std_dev_j = np.std([j for j in range(sz[1])])
        kern = np.zeros(sz)
        for i in range(sz[0]):
            for j in range(sz[1]):
                dist = (i - sz[0]/2)**2 + (j - sz[1]/2)**2
                kern[i,j] = np.exp(-dist / ((std_dev_i**2+ std_dev_j**2)* intensity))
                
                # Normailize the Gaussian Filter
                energy = np.sqrt(np.sum(np.power(kern,2),axis = 1))
        return kern / energy
    elif(len(sz) == 3):
        pass # add if needed
        

def get_horz_edge_kern():
    Hx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = np.float32)
    return Hx

def get_vert_edge_kern():    
    Hy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype = np.float32)
    return Hy.T
        













        
        
        