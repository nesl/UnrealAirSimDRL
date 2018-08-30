# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 00:56:52 2018

@author: natsn
"""

# use open cv to create point cloud from depth image.
import airsim
#from airsim import client
import cv2
import time
from airsim.types import Vector3r
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  axes3d, Axes3D
############################################
########## This is work in progress! #######
############################################

# file will be saved in PythonClient folder (i.e. same folder as script)
# point cloud ASCII format, use viewers like CloudCompare http://www.danielgm.net/cc/ or see http://www.geonext.nl/wp-content/uploads/2014/05/Point-Cloud-Viewers.pdf
outputFile = "cloud.asc" 
color = (0,255,0)
rgb = "%d %d %d" % color
projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                              [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                              [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                              [0.000000000, 0.000000000, -10.0000000, 0.000000000]])


# Connect to the AirSim simulator and begin:
print('Initializing Car Client')
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)
print('Initialization DONE!')

print("Setting Camera Views")
orien = Vector3r(0, 0, 0)
client.simSetCameraOrientation("3", orien) #radians
orien = Vector3r(0, 0, -np.pi/2)
client.simSetCameraOrientation("1", orien)
orien = Vector3r(0, 0, np.pi/2)
client.simSetCameraOrientation("2", orien)
orien = Vector3r(0, 0, np.pi/2)
# Reset Collion Flags
print("Setting Camera Views DONE!")



while True:
    print("Making a Point Cloud!")
    images = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, False, False), # Front Center
            airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, False, False), # Front Right
            airsim.ImageRequest("2", airsim.ImageType.DepthPlanner, False, False)]) # Front Left
    img1d_FC = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
    img_depth_FC = np.array(img1d_FC.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
    img_depth_FC = img_depth_FC[:,:,0:3]
    
    img1d_FR = np.fromstring(images[1].image_data_uint8, dtype=np.uint8) 
    img_depth_FR = np.array(img1d_FR.reshape(images[1].height, images[1].width, 4), dtype = np.uint8)
    img_depth_FR = img_depth_FR[:,:,0:3]
    
    img1d_FL = np.fromstring(images[2].image_data_uint8, dtype=np.uint8) 
    img_depth_FL = np.array(img1d_FL.reshape(images[2].height, images[2].width, 4), dtype = np.uint8)
    img_depth_FL = img_depth_FL[:,:,0:3]
                    
    #img_rgb_FC = img_rgba_FC[:,:,0:3]
    #noIdea = client.simGetImage("0", airsim.ImageType.DepthPlanner)`
    
    #print("Shape: ", img.shape)
    gray = np.mean(img_depth_FC, axis = 2, dtype = np.uint8)
    Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
    print(Image3D.shape, type(Image3D))
    print(Image3D)
    Image3D_edit = Image3D
    plt.close()
    fig = plt.figure(1, figsize= (4,4))
    ax = Axes3D(fig)
    ax.scatter(np.random.rand(10), np.random.rand(10), np.random.rand(10))
    
    
    
    plt.figure(2)
    plt.imshow(img_depth_FC, cmap = 'gray')
    plt.figure(3)
    plt.imshow(img_depth_FR, cmap = 'gray')
    plt.figure(4)
    plt.imshow(img_depth_FL, cmap = 'gray')
    plt.show()
    time.sleep(2)

    key = cv2.waitKey(5) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        print("Goodbye!")
        break;
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        