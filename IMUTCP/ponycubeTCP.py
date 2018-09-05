# Rotate a cube with a quaternion
# Demo program
# Pat Hickey, 27 Dec 10
# This code is in the public domain.

import pygame
import pygame.draw
import pygame.time
from math import sin, cos, acos
from euclid import *
import tkinter as tk
from tkinter import *
from tkinter.ttk import Label, Frame, Entry, Notebook, Combobox
import os

import numpy

import IMUTCPClient

class Screen (object):
    def __init__(self,x=320,y=280,scale=1):
        self.i = pygame.display.set_mode((x,y))
        self.originx = self.i.get_width() / 2
        self.originy = self.i.get_height() / 2
        self.scale = scale

    def project(self,v):
        assert isinstance(v,Vector3)
        x = v.x * self.scale + self.originx
        y = v.y * self.scale + self.originy
        return (x,y)
    def depth(self,v):
        assert isinstance(v,Vector3)
        return v.z

class PrespectiveScreen(Screen):
    # the xy projection and depth functions are really an orthonormal space
    # but here i just approximated it with decimals to keep it quick n dirty
    def project(self,v):
        assert isinstance(v,Vector3)
        x = ((v.x*0.957) + (v.z*0.287)) * self.scale + self.originx
        y = ((v.y*0.957) + (v.z*0.287)) * self.scale + self.originy
        return (x,y)
    def depth(self,v):
        assert isinstance(v,Vector3)
        z = (v.z*0.9205) - (v.x*0.276) - (v.y*0.276)
        return z

class Side (object):
    def __init__(self,a,b,c,d,color=(50,0,0)):
        assert isinstance(a,Vector3)
        assert isinstance(b,Vector3)
        assert isinstance(c,Vector3)
        assert isinstance(d,Vector3)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.color = color

    def centroid(self):
        return ( self.a + self.b + self.c + self.d ) / 4

    def draw(self,screen):
        assert isinstance(screen,Screen)
        s = [ screen.project(self.a)
            , screen.project(self.b)
            , screen.project(self.c)
            , screen.project(self.d)
            ]
        pygame.draw.polygon(screen.i,self.color,s)

    def erase(self,screen,clear_color = (0,0,0)):
        c = self.color
        self.color = clear_color
        self.draw(screen)
        self.color = c

class Edge (object):
    def __init__(self,a,b,color=(0,0,255)):
        assert isinstance(a,Vector3)
        assert isinstance(b,Vector3)
        self.a = a
        self.b = b
        self.color = color
    def centroid(self):
        return (self.a + self.b) / 2
    def draw(self,screen):
        assert isinstance(screen,Screen) 
        aa = screen.project(self.a)
        bb = screen.project(self.b)
        pygame.draw.line(screen.i, self.color, aa,bb)
    def erase(self,screen,clear_color = (0,0,0)):
        c = self.color
        self.color = clear_color
        self.draw(screen)
        self.color = c
        
        

class Cube (object):
    def __init__(self,a=10,b=10,c=10):
        self.a = a
        self.b = b
        self.c = c
        self.pts = [ Vector3(-a,b,c)  , Vector3(a,b,c)
                   , Vector3(a,-b,c)  , Vector3(-a,-b,c)
                   , Vector3(-a,b,-c) , Vector3(a,b,-c)
                   , Vector3(a,-b,-c) , Vector3(-a,-b,-c) ] 

    def origin(self):
        """ reset self.pts to the origin, so we can give them a new rotation """
        a = self.a; b = self.b; c = self.c
        self.pts = [ Vector3(-a,b,c)  , Vector3(a,b,c)
                   , Vector3(a,-b,c)  , Vector3(-a,-b,c)
                   , Vector3(-a,b,-c) , Vector3(a,b,-c)
                   , Vector3(a,-b,-c) , Vector3(-a,-b,-c) ] 

    def sides(self):
        """ each side is a Side object of a certain color """
        # leftright  = (80,80,150) # color
        # topbot     = (30,30,150)
        # frontback  = (0,0,150)
        one =   (255, 0, 0)
        two =   (0, 255, 0)
        three = (0, 0, 255)
        four =  (255, 255, 0)
        five =  (0, 255, 255)
        six =   (255, 0, 255)
        a, b, c, d, e, f, g, h = self.pts
        sides = [ Side( a, b, c, d, one)   #  front
                , Side( e, f, g, h, two)   #  back
                , Side( a, e, f, b, three) #  bottom
                , Side( b, f, g, c, four)  # right
                , Side( c, g, h, d, five)  # top
                , Side( d, h, e, a, six)   # left
                ]
        return sides

    def edges(self):
        """ each edge is drawn as well """
        ec         = (0,0,255) # color
        a, b, c, d, e, f, g, h = self.pts
        edges = [ Edge(a,b,ec), Edge(b,c,ec), Edge(c,d,ec), Edge(d,a,ec)
                , Edge(e,f,ec), Edge(f,g,ec), Edge(g,h,ec), Edge(h,e,ec)
                , Edge(a,e,ec), Edge(b,f,ec), Edge(c,g,ec), Edge(d,h,ec)
                ]
        return edges

    def erase(self,screen):
        """ erase object at present rotation (last one drawn to screen) """
        assert isinstance(screen,Screen)
        sides = self.sides()
        edges = self.edges()
        erasables = sides + edges
        [ s.erase(screen) for s in erasables]

    def draw(self,screen,q=Quaternion(1,0,0,0)):
        """ draw object at given rotation """
        assert isinstance(screen,Screen)
        self.origin()
        self.rotate(q)
        sides = self.sides()
        edges = self.edges()
        drawables = sides + edges
        drawables.sort(key=lambda s: screen.depth(s.centroid()))
        [ s.draw(screen) for s in drawables ]

    def rotate(self,q):
        assert isinstance(q,Quaternion)
        R = q.get_matrix()
        self.pts = [R*p for p in self.pts]


def imu_visualization():
    #TKINTER setup
    root = tk.Tk()
    root.title = 'Imu visualization'
    nb = Notebook(root)
    #Grid the notebook
    nb.grid(row = 0, column = 0)
    
    #initialize the frame
    embed = tk.Frame(root, width=580, height = 500)
    #if the notebook is grided, no need to pack, if not, use the following one line of code
    #embed.pack()

    nb.add(embed, text="imuviz tab")

    



    #Tell pygame's SDL window which window ID to use
    os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
    #show the window so it's assigned an ID
    root.update()

    label_yaw = Label(embed, text = "Yaw:")
    label_pitch = Label(embed, text = "Pitch:")
    label_roll = Label(embed, text = "Roll:")

    label_yaw.place(x=500,y = 0)
    label_pitch.place(x=500, y = 50)
    label_roll.place(x=500, y = 100)

    entry_yaw = Entry(embed, text = "Yaw:")
    entry_pitch = Entry(embed, text = "Pitch:")
    entry_roll = Entry(embed, text = "Roll:")

    entry_yaw.place(x = 500, y = 25)
    entry_pitch.place(x = 500, y = 75)
    entry_roll.place(x = 500, y = 125)

    #PYGAME
    pygame.init()
    screen = Screen(500,500,scale=1.5)
    cube = Cube(40,30,60)
    
    # test the ponycube
    q = Quaternion(1,0,0,0)
    incr = Quaternion(0.96,0.01,0.01,0).normalized()

    #initialize client
    imuc = IMUTCPClient.IMUTCPClient()
    
    while True:
        #recieve IMU data
        control = imuc.recv_imu_update() 
        if control is not None: 
            print(control) 
            
            entry_yaw.delete(0,tk.END)
            entry_yaw.insert(0, str(control["yaw"]))
            entry_pitch.delete(0,tk.END)
            entry_pitch.insert(0, str(control["pitch"]))
            entry_roll.delete(0,tk.END)
            entry_roll.insert(0, str(control["roll"]))
             
            # #test the entry using random numbers
            # entry_yaw.delete(0,tk.END)
            # entry_yaw.insert(0, str(round(numpy.random.randn(),6)))
            # entry_pitch.delete(0,tk.END)
            # entry_pitch.insert(0, str(round(numpy.random.randn(),6)))
            # entry_roll.delete(0,tk.END)
            # entry_roll.insert(0, str(round(numpy.random.randn(),6)))
            
            q = Quaternion(control["quatW"], control["quatX"], control["quatY"], control["quatZ"]).normalized()

            # #test the ponycube
            # q = q * incr

            cube.draw(screen,q)
            event = pygame.event.poll()
            pygame.display.flip()
            pygame.time.delay(50) 
            cube.erase(screen)
            #TKINTER
            root.update()

if __name__ =='__main__':

    imu_visualization()
  