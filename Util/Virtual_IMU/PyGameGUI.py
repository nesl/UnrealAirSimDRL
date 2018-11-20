# -*- coding: utf-8 -*-
# Rotate a cube with a quaternion
# Demo program
# Pat Hickey, 27 Dec 10
# This code is in the public domain.
import sys, os
import time
import pygame
import pygame.draw
import pygame.time
import numpy as np
import scipy.signal as signal
import scipy.linalg as la
from math import sin, cos, acos
import SO3Rotation as SO3
import IMUListener
from euclid import Vector3, Quaternion
from EuclidObjects import Cube, Screen, Grid, PerspectiveScreen
sys.path.append("/home/natsubuntu/Desktop/UnrealAirSimDRL/Util/")
import XboxListenerLinux


class PyGameGUI(object):

    def __init__(self,
                 is2D = False,
                 isGrid = True,
                 screensize = [480, 400],
                 control_method = 'test',     #'ue'/'xbox'/'test'
                 cubeMass = 2,
                 cubeInertial = 0.5,
                 cubeSize = [10,10,10],
                 node_num = 5,
                 gap = 20,
                 relative_posi = Vector3(0,0,0),
                 x_proj_angle = 30,
                 y_proj_angle = 15):

        self.is2D = is2D
        self.isGrid = isGrid
        self.screensize = screensize
        self.control_method = control_method
        self.cubeMass = cubeMass
        self.cubeInertial = cubeInertial
        self.cubeSize = cubeSize
        self.node_num = node_num
        self.gap = gap
        self.relative_posi = relative_posi
        self.x_proj_angle = x_proj_angle
        self.y_proj_angle = y_proj_angle
        
        pygame.init()

        if self.is2D:
            self.screen = Screen(self.screensize[0],self.screensize[1],scale=1.5)
        else:
            self.screen = PerspectiveScreen(self.screensize[0],self.screensize[1],1.5,self.x_proj_angle,self.y_proj_angle)

        self.cube = Cube(self.cubeSize[0],self.cubeSize[1],self.cubeSize[2])
        self.grid = Grid()
    
    def run(self):
        tic = time.time()
        if self.isGrid:
            self.grid.draw(self.screen)
                    
        if self.control_method == 'test':
            q = Quaternion(1,0,0,0) # Unit Quaternion
            incr = Quaternion(0.96,0.01,0.01,0).normalized()
            count = 0

            while 1:
                q = q*incr
                if self.isGrid:
                    self.grid.draw(self.screen)
                self.cube.draw(self.screen, q, Vector3(-count,count,count))       
                event = pygame.event.poll()
                if event.type == pygame.QUIT \
                    or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    break
                pygame.display.flip()
                pygame.time.delay(100) 
                self.cube.erase(self.screen)
                count-=1

        # First Order Rotation/Translation Model
        elif self.control_method == 'xbox':
            sample_rate = .0005
            #xbl = XboxListenerWindows.XBoxListener(delay_sec)
            xbl = XboxListenerLinux.XBoxListener(sample_rate)
            xbl.init()

            rot_yaw_rate = 0.0
            rot_pitch_rate = 0.0
            rot_roll_rate = 0.0
            vel_rate = 0.0
            TOP_RATE = 2/3*np.pi
            rpy = Vector3(0, 0, 0)
            rpy_t = Vector3(0, 0, 0)
            xyz = Vector3(self.cube.a, self.cube.b, self.cube.c)
            xyz_t = Vector3(self.cube.a, self.cube.b, self.cube.c)
            d_xyz = Vector3(0, 0, 0)
            cur_pose = Quaternion(1, 0, 0, 0)
            d_rpy = Vector3(0, 0, 0)

            # Need To implement full quaternion dynamics
            quat = np.matrix([1,0,0,0]).T
            pos = np.matrix([0,0,0]).T
            vel = np.matrix([0,0,0]).T
            R = np.matrix(np.eye(4))
            u = np.array([0,0,0,0]) # angular velocity and acceleration
            # 1) With input form rotation matrix 
            # 2) Apply rotation matrix to position and velocity updates
            # 3) Apply quaternion update formula for orientation

            t = time.time()
            while True:
                dt = 0.05  # Desired
                tnow = time.time()
                if tnow - t < dt:
                    time.sleep(dt - (tnow - t))
                else:
                    dt = tnow - t
                t = time.time()

                control = xbl.get()
                if control is not None: # new update
                    rot_yaw_rate = float(control['leftX'])
                    rot_pitch_rate = float(control['rightY'])
                    rot_roll_rate = float(control['rightX'])
                    acc_rate = float(control['leftY'])
                else:
                    rot_rate_yaw = 0
                    rot_pitch_rate = 0
                    rot_roll_rate = 0
                    acc_rate = 0
                # Need To implement full quaternion dynamics
                u[0] = TOP_RATE*rot_roll_rate
                u[1] = TOP_RATE*rot_pitch_rate
                u[2] = TOP_RATE*rot_yaw_rate
                u[3] = 40*acc_rate
                acc = np.matrix([0,0,u[3]]).T
                # Apply Quaternion Update:
                Omega = SO3.get_omega_matrix(u[0],u[1],u[2])
                #quat = quat + 1/2*dt*Omega*quat # ZOH -- qdot = .5*Omega*quat
                quat = la.expm(Omega*dt/2.0)*quat
                R = SO3.quaternion_to_rotation_matrix(np.array(quat))
                pos = pos + vel*dt + 1/2*dt**2*R*acc
                vel = vel + dt*R*acc

                xyz = Vector3(pos[0],pos[1],pos[2])
                pose = Quaternion(quat[0],quat[1],quat[2],quat[3])

                #print("Current Position", pos)
                #print("Current Pose", quat)
                if self.isGrid:
                    self.grid.draw(self.screen)
                # Draw Updates
                self.cube.draw(self.screen, pose, xyz)
                
                event = pygame.event.poll()
                if event.type == pygame.QUIT \
                    or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    break
                pygame.display.flip()
                pygame.time.delay(1) #Fast Update
                self.cube.erase(self.screen)
                #for c in cubes:
                #    c.erase(self.screen)

        elif self.control_method == 'imu':
            delay_sec = .05
            imuc = IMUListener.IMUListener(delay_sec)
            imuc.init()
            while True:
                control = imuc.get()
                print(control)
                if control is not None:
                    q = Quaternion(control["quatW"], control["quatX"], control["quatY"], control["quatZ"]).normalized()
                    posi = Vector3(0,0,0)
                    if self.isGrid:
                        self.grid.draw(self.screen)
                    self.cube.draw(self.screen,q,posi)       
                    event = pygame.event.poll()
                    if event.type == pygame.QUIT \
                        or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        break
                    pygame.display.flip()
                    pygame.time.delay(20) 
                    self.cube.erase(self.screen)
            
        else:
            print('invalid control input!')


    def test_imu(self):
         delay_sec = .05
         imuc = IMUListener.IMUListener(delay_sec)
         imuc.init()
         while True:
            control = imuc.get()
            print(control)
            if control is not None:
                q = Quaternion(control["quatW"], control["quatX"],
                                control["quatY"], control["quatZ"]).normalized()
                posi = Vector3(0, 0, 0)
                if self.isGrid:
                    self.grid.draw(self.screen)
                self.cube.draw(self.screen, q, posi)
                event = pygame.event.poll()
                if event.type == pygame.QUIT \
                        or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    break
                pygame.display.flip()
                pygame.time.delay(20)
                self.cube.erase(self.screen)

    def test_xbox(self):
        tic = time.time()
        if self.isGrid:
            self.grid.draw(self.screen)
        # First Order Rotation/Translation Model
        sample_rate = .0005
        #xbl = XboxListenerWindows.XBoxListener(delay_sec)
        xbl = XboxListenerLinux.XBoxListener(sample_rate)
        xbl.init()

        rot_yaw_rate = 0.0
        rot_pitch_rate = 0.0
        rot_roll_rate = 0.0
        vel_rate = 0.0
        TOP_RATE = 2/3*np.pi
        rpy = Vector3(0, 0, 0)
        rpy_t = Vector3(0, 0, 0)
        xyz = Vector3(self.cube.a, self.cube.b, self.cube.c)
        xyz_t = Vector3(self.cube.a, self.cube.b, self.cube.c)
        d_xyz = Vector3(0, 0, 0)
        cur_pose = Quaternion(1, 0, 0, 0)
        d_rpy = Vector3(0, 0, 0)

        # Need To implement full quaternion dynamics
        quat = np.matrix([1, 0, 0, 0]).T
        pos = np.matrix([0, 0, 0]).T
        vel = np.matrix([0, 0, 0]).T
        R = np.matrix(np.eye(4))
        u = np.array([0, 0, 0, 0])  # angular velocity and acceleration
        # 1) With input form rotation matrix
        # 2) Apply rotation matrix to position and velocity updates
        # 3) Apply quaternion update formula for orientation

        t = time.time()
        while True:
            dt = 0.05  # Desired
            tnow = time.time()
            if tnow - t < dt:
                time.sleep(dt - (tnow - t))
            else:
                dt = tnow - t
            t = time.time()

            control = xbl.get()
            if control is not None:  # new update
                rot_yaw_rate = float(control['leftX'])
                rot_pitch_rate = float(control['rightY'])
                rot_roll_rate = float(control['rightX'])
                acc_rate = float(control['leftY'])
            else:
                rot_rate_yaw = 0
                rot_pitch_rate = 0
                rot_roll_rate = 0
                acc_rate = 0
            # Implementing quaternion dynamics
            u[0] = TOP_RATE*rot_roll_rate
            u[1] = TOP_RATE*rot_pitch_rate
            u[2] = TOP_RATE*rot_yaw_rate
            u[3] = 40*acc_rate
            acc = np.matrix([0, 0, u[3]]).T
            # Apply Quaternion Update:
            Omega = SO3.get_omega_matrix(u[0], u[1], u[2])
            #quat = quat + 1/2*dt*Omega*quat # ZOH -- qdot = .5*Omega*quat
            quat = la.expm(Omega*dt/2.0)*quat
            R = SO3.quaternion_to_rotation_matrix(np.array(quat))
            pos = pos + vel*dt + 1/2*dt**2*R*acc
            vel = vel + dt*R*acc

            xyz = Vector3(pos[0], pos[1], pos[2])
            pose = Quaternion(quat[0], quat[1], quat[2], quat[3])

            #print("Current Position", pos)
            #print("Current Pose", quat)
            if self.isGrid:
                self.grid.draw(self.screen)
            # Draw Updates
            self.cube.draw(self.screen, pose, xyz)

            event = pygame.event.poll()
            if event.type == pygame.QUIT \
                    or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                break
            pygame.display.flip()
            pygame.time.delay(1)  # Fast Update 1 ms
            self.cube.erase(self.screen)

        



      

if __name__ == "__main__":
    gui = PyGameGUI(control_method = "xbox")
    gui.run()
