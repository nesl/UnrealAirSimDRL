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
from math import sin, cos, acos
import SO3Rotation as SO3
import IMUListener
from euclid import Vector3, Quaternion
from EuclidObjects import Cube, Screen, Grid, PerspectiveScreen
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")
import XboxListenerWindows


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
            q = Quaternion(1,0,0,0)
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

        elif self.control_method == 'xbox':
            delay_sec = .05
            xbl = XboxListenerWindows.XBoxListener(delay_sec)
            xbl.init()
            rotation_thrust_rates = {'ly': 0, 'lx': 0, 'rx': 0, 'ry': 0} # thurst, yaw, pitch, roll
            last_control_rate = 0
            last_control_axis = 'ly'
            self.rtr = 0
            last_quat = SO3.euler_angles_to_quaternion(0, 0, 0)
            while True:
                control = xbl.get()
                dt = time.time() - tic
                tic = time.time()
                TOP_RATE = 2*np.pi
                print("RTR", rotation_thrust_rates)
                if control is not None: # new update
                    # convert the new angles into a quaternion
                    rotation_thrust_rates[control['LA']] += control['LAV']*TOP_RATE*dt
                    last_control_rate = control['LAV']*TOP_RATE*dt
                    last_control_axis = control['LA']
                elif np.abs(last_control_rate) > .03:
                    rotation_thrust_rates[last_control_axis] += last_control_rate
                
                #print("RTR", rotation_thrust_rates)
                #print(control)
                quat = SO3.euler_angles_to_quaternion(rotation_thrust_rates['lx'], rotation_thrust_rates['rx'], rotation_thrust_rates['ry'])
                rot = SO3.quaternion_to_rotation_matrix(quat - last_quat)
                self.rtr += rotation_thrust_rates['ly'] - self.rtr
                posz = np.matrix([0,0, self.rtr])
                #print(rot, q, self.rtr, posz)
                vectors = [np.array(np.random.randn(3,1))*150 for i in range(20)]
                posz = rot*posz.T
                posz = Vector3(posz[0], posz[1], posz[2])
                q = Quaternion(quat[0], quat[1], quat[2], quat[3]).normalized()
                cubes = []
                for v in vectors:
                    cube = Cube(self.cubeSize[0]/3,self.cubeSize[1]/3,self.cubeSize[2]/3)
                    p = Vector3(v[0], v[1], v[2])
                    cube.draw(self.screen,q,p)
                    cubes.append(cube)
                if self.isGrid:
                    self.grid.draw(self.screen)
                self.cube.draw(self.screen,q,posz)       
                event = pygame.event.poll()
                if event.type == pygame.QUIT \
                    or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    break
                pygame.display.flip()
                pygame.time.delay(int(delay_sec*50))
                self.cube.erase(self.screen)
                for c in cubes:
                    c.erase(self.screen)
                last_quat = quat

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




      

if __name__ == "__main__":
    gui = PyGameGUI(control_method = "xbox")
    gui.run()