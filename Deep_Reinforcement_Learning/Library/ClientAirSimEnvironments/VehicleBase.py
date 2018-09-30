# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:35:50 2018

@author: natsn
"""

import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\..\\Util")
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..\\..\\..\\Util\\Virtual_IMU")
sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "\\..")
import SO3Rotation
from airsim import client

# Image Classifications
class AirSimImageTypes:
    Scene = 0
    DepthPlanner = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6
    Infrared = 7
    RGBA = 8
    RGBA_Normal = 9
    Scene_Normal = 10 
    Gray = 11
    Gray_Normal = 12
    
    img_types_vals = {"Scene": Scene,
                 "DepthPlanner": DepthPlanner,
                 "DepthPerspective": DepthPerspective,
                 "DepthVis": DepthVis,
                 "DisparityNormalized": DisparityNormalized,
                 "Segmentation": Segmentation,
                 "SurfaceNormals": SurfaceNormals,
                 "Infrared": Infrared,
                 "RGBA": RGBA,
                 "RGBA_Normal": RGBA_Normal,
                 "Scene_Normal": Scene_Normal,
                 "Gray": Gray,
                 "Gray_Normal": Gray_Normal}
    
    img_vals_types = { Scene:"Scene",
                 DepthPlanner: "DepthPlanner",
                 DepthPerspective:"DepthPerspective",
                 DepthVis: "DepthVis",
                 DisparityNormalized: "DisparityNormalized",
                 Segmentation: "Segmentation",
                 SurfaceNormals: "SurfaceNormals",
                 Infrared: "Infrared",
                 RGBA: "RGBA",
                 RGBA_Normal: "RGBA_Normal",
                 Scene_Normal: "Scene_Normal",
                 Gray:  "Gray",
                 Gray_Normal: "Gray_Normal"}
   
    
    img_types_vals = {"Scene": 3,
                 "DepthPlanner": 4,
                 "DepthPerspective": 4,
                 "DepthVis": 4,
                 "DisparityNormalized": 4,
                 "Segmentation": 4,
                 "SurfaceNormals": 3,
                 "Infrared": 3,
                 "RGBA": 4,
                 "RGBA_Normal": 4,
                 "Scene_Normal": 3,
                 "Gray": 1,
                 "Gray_Normal": 1}
    

class AirSimImageRetrievals:
    
    def __init__(self):
        self.ImageTypes = AirSimImageTypes
    
    def retrieval_4d(self, client, camera_name, img_arg1 = True, img_arg2 = False):
        images = client.simGetImages([client.ImageRequest(camera_name, client.ImageType.Scene, img_arg1, img_arg2)]) # Front Center
        img = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
        img = np.array(img.reshape(images[0].height, images[0].width, 4), dtype = np.uint8)
        return img
    
    def retrieval_3d(self, client, camera_name, img_arg1 = True, img_arg2 = False):
        images = client.simGetImages([client.ImageRequest(camera_name, client.ImageType.Scene, img_arg1, img_arg2)]) # Front Center
        img = np.fromstring(images[0].image_data_uint8, dtype=np.uint8) 
        img = np.array(img.reshape(images[0].height, images[0].width, 3), dtype = np.uint8)
        return img
    
    def retrieval_4d_to_3d(self, client, camera_name, img_arg1 = False, img_arg2 = False):
        img = self.retrieval_4d(client, camera_name, img_arg1, img_arg2)
        return img[:,:,0:3]
    
    def convert_3d_to_1d(self, img):
        img = np.mean(img, axis = 2)
        return img.mean(axis = 2)
        
    def normalize(img):
        return (img - img.mean(axis = 2)) / img.std(axis = 2)
    
    # takes values to 0-1
    def squeeze(img):
        return np.array(img / 255.0, dtype = np.float32)
    
    def retrieval(self, client, type_of_retrieval):
        if type_of_retrieval == self.ImageTypes.Scene:
            return self.retrieval_4d_to_3d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.DepthPlanner:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.DepthPerspective:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.DepthVis:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.DisparityNormalized:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.Segmentation:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.SurfaceNormals:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.Infrared:
            return self.retrieval_4d(client, type_of_retrieval)
        elif type_of_retrieval == self.ImageTypes.RGBA:
            return self.retrieval_4d(client, self.ImageTypes.Scene)
        elif type_of_retrieval == self.ImageTypes.RGBA_Normal:
            img = self.retrieval_4d(client, self.ImageTypes.Scene)
            return self.normalize(img)
        elif type_of_retrieval == self.ImageTypes.Scene_Normal:
            img = self.retrieval_4d_to_3d(client, self.ImagesTypes.Scene)
            return self.normalize(img)
        elif type_of_retrieval == self.ImageTypes.Gray:
            img = self.retrieval_4d_to_3d(client, self.ImageTypes.Scene)
            return self.convert_3d_to_1d(img)
        elif type_of_retrieval == self.ImageTypes.Gray_Normal:
            img = self.retrieval_4d_to_3d(client, type_of_retrieval)
            img = self.convert_3d_to_1d(img)
            return self.squeeze(img)
        else:
            print("Unknown Retrieval Type")
            

class AirSimVehicle:
    
    # load up the dictionary with all image modes you want from each view
    # Image modes are defined in the AirSimImageTypes Class
    # Input the angle at which the cameras should turn
    def __init__(self, isDrone = True,
                 VehicleCamerasModesAndViews = {"front_center": [np.array([]), (0,0,0)],
                     "front_right": [np.array([]), (0,0,0)],
                     "front_left": [np.array([]), (0,0,0)],
                     "fpv": [np.array([]), (0,0,0)],
                     "back_center": [np.array([]), (0,0,0)]} ):
        # Helper Classes
        self.imgTypes = AirSimImageTypes
        self.imgRetrievals = AirSimImageRetrievals
        
        # Class Helpers
        self.VehicleCamerasModesAndViews = VehicleCamerasModesAndViews
        self.isDrone = isDrone
        
        # Setup Img Retrieval
        self.initialize_vehicle_type()
        self.set_camera_angles()
        
    # Initialize Vehicle Type
    def initialize_vehicle_type(self):
        if not self.isDrone:
            # Connect to the AirSim simulator and begin:
            print('Initializing Car Client')
            self.client = client.CarClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            print('Initialization DONE!')
        else:
            print('Initializing Drone Client')
            self.client = client.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            print('Initialization DONE!')
    
    # Set the active camera views on the vehicle
    # Pass in a dictionary of each vehicle view with the ypr you want it at (DEGREES)
    # Angles should be a tuple of yaw pitch and roll
    def set_camera_angles(self, views_angles = None):
        print("Setting Camera Views")
        if views_angles is None: # Initialize VehicleImageViews
            for key, values in self.VehicleImageViewsAndModes:
                if len(self.VehicleImageViewsAndModes[key][0]) > 0:
                    quat = SO3Rotation.euler_angles_to_quaternion(values[1][0], values[1][1], values[1][2])
                    self.client.simSetCameraOrientation("0", quat) # initialize every view of the vehicle thats set
                    print(key, "set!")
        else: # Change Camera view on fly
            assert type(views_angles) == dict
            for v,a in views_angles:
                quat = SO3Rotation.euler_angles_to_quaternion(a[0], a[1], a[2])
                self.client.simSetCameraOrientation("0", quat) # this can be used to change the camera views on the fly
                print(v, "set!")
        print("Setting Camera Views DONE!")
        
    # Image Retrieval
    def images_retrieval(self):
        # Construct the Images State Vector
        # Order is Front Center, Front Right, Front Left
        retrieval_dic = {}
        #tic = time.time()
        for key, values in self.VehicleCamerasModesAndViews:
            if len(self.VehicleImageViewsAndModes[key][0]) != 0:
                retrieval_dic[key] = []
                for v in values[0]:
                    retrieval_dic[key].append({self.imgTypes.img_vals_types[v], self.imgRetrievals.retrieval(self.client, v)})
        return retrieval_dic
     
    def stack_camera_feeds(retrieval_dic):
        stacks = []
        for ret in retrieval_dic:
            tmp = None
            for r,v in retrieval_dic[ret]:
                if tmp is None:
                    tmp = v
                else:
                    tmp = np.dstack((tmp,v))
            stacks.append(tmp)
        return stacks
    
    def all_feeds_to_single_tensor(stacks):
        tmp = None
        for s in stacks:
            if tmp is None:
                tmp = s
            else:
                tmp = np.dstack((tmp,s))
        return tmp
