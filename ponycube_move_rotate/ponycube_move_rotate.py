# Rotate a cube with a quaternion
# Demo program
# Pat Hickey, 27 Dec 10
# This code is in the public domain.

import pygame
import pygame.draw
import pygame.time
from math import sin, cos, acos
from euclid import *

#import IMUTCPClient

class Screen (object):
    def __init__(self,x=320,y=280,scale=1):
        self.i = pygame.display.set_mode((x,y))
        self.i.fill([255,255,255])
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

class PerspectiveScreen(Screen):
    # the xy projection and depth functions are really an orthonormal space
    # but here i just approximated it with decimals to keep it quick n dirty
    def project(self,v):
        assert isinstance(v,Vector3)
        # x = ((v.x*0.957) + (v.z*0.287)) * self.scale + self.originx
        # y = ((v.y*0.957) + (v.z*0.287)) * self.scale + self.originy
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

    def erase(self,screen,clear_color = (255,255,255)):
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
    def erase(self,screen,clear_color = (145,168,209)):
        c = self.color
        self.color = clear_color
        self.draw(screen)
        self.color = c

               

class Cube (object):
    def __init__(self,a=10,b=10,c=10):
        self.a = a
        self.b = b
        self.c = c
        self.pts = [ Vector3(-a,b,c), Vector3(a,b,c)
                   , Vector3(a,-b,c), Vector3(-a,-b,c)
                   , Vector3(-a,b,-c), Vector3(a,b,-c)
                   , Vector3(a,-b,-c), Vector3(-a,-b,-c)] 

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
        one =   (145, 168, 209)
        two =   (247, 202, 201)
        three = (247, 202, 201)
        four =  (247, 202, 201)
        five =  (247, 202, 201)
        six =   (247, 202, 201)
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

    def draw(self,screen,q=Quaternion(1,0,0,0),centre_pos=Vector3(0,0,0)):
        """ draw object at given rotation """
        assert isinstance(screen,Screen)
        self.origin()
        #self.rotate(q)
        self.rotate_move(q,centre_pos)
        sides = self.sides()
        edges = self.edges()
        drawables = sides + edges
        drawables.sort(key=lambda s: screen.depth(s.centroid()))
        [ s.draw(screen) for s in drawables ]


    def rotate(self,q):
        assert isinstance(q,Quaternion)
        R = q.get_matrix()
        self.pts = [R*p for p in self.pts]

    def rotate_move(self,q,centre_pos):
        assert isinstance(q,Quaternion)
        R = q.get_matrix()
        self.pts = [R*p for p in self.pts]
        self.pts = [p+centre_pos for p in self.pts]
  


# y axis should be minus
class Grid(object):
    def __init__(self, node_num=5, gap=20):
        self.node_num = node_num
        self.gap = gap


        self.origin = Vector3(0,0,0)
        self.grid_pts_x_axis = []
        self.grid_pts_y_axis = []
        self.grid_pts_z_axis = []

        for i in range(self.node_num):
            self.grid_pts_x_axis.append(Vector3((i+1)*self.gap, 0, 0))
            self.grid_pts_y_axis.append(Vector3(0, -(i+1)*self.gap, 0))
            self.grid_pts_z_axis.append(Vector3(0, 0, (i+1)*self.gap))

        self.grid_pts_x_xy = [v+Vector3(0, -self.node_num*self.gap, 0) for v in self.grid_pts_x_axis]        
        self.grid_pts_x_xz = [v+Vector3(0, 0, self.node_num*self.gap) for v in self.grid_pts_x_axis]
        self.grid_pts_y_yx = [v+Vector3(self.node_num*self.gap, 0, 0) for v in self.grid_pts_y_axis]
        self.grid_pts_y_yz = [v+Vector3(0, 0, self.node_num*self.gap) for v in self.grid_pts_y_axis]
        self.grid_pts_z_zx = [v+Vector3(self.node_num*self.gap, 0, 0) for v in self.grid_pts_z_axis]
        self.grid_pts_z_zy = [v+Vector3(0, -self.node_num*self.gap, 0) for v in self.grid_pts_z_axis]
        

    def lines(self):
        ec         = (0,0,0) # color
        x = self.grid_pts_x_axis
        y = self.grid_pts_y_axis
        z = self.grid_pts_z_axis
        x_xy = self.grid_pts_x_xy
        x_xz = self.grid_pts_x_xz
        y_yx = self.grid_pts_y_yx
        y_yz = self.grid_pts_y_yz
        z_zx = self.grid_pts_z_zx
        z_zy = self.grid_pts_z_zy

        axes = [ Edge(self.origin,x[-1],ec), Edge(self.origin,y[-1],ec), Edge(self.origin,z[-1],ec)]
        lines = []
        for i in range(len(x)):
            lines.append(Edge(x[i],x_xy[i],ec))
            lines.append(Edge(x[i],x_xz[i],ec))
            lines.append(Edge(y[i],y_yx[i],ec))
            lines.append(Edge(y[i],y_yz[i],ec))
            lines.append(Edge(z[i],z_zx[i],ec))
            lines.append(Edge(z[i],z_zy[i],ec))
                       
        
        return axes + lines

    def draw(self,screen):
        drawables = self.lines()
        drawables.sort(key=lambda s: screen.depth(s.centroid()))
        [ s.draw(screen) for s in drawables ]



class PygameImuGui(object):

    def __init__(self,
                 is2D = False,
                 isGrid = True,
                 screensize = [480, 400],
                 control_input = 'test',     #'ue'/'xbox'/'test'
                 cubeMass = 2,
                 cubeInertial = 0.5,
                 cubeSize = [10,10,10],
                 node_num = 5,
                 gap = 20,
                 relative_posi = Vector3(0,0,0)):

        self.is2D = is2D
        self.isGrid = isGrid
        self.screensize = screensize
        self.control_input = control_input
        self.cubeMass = cubeMass
        self.cubeInertial = cubeInertial
        self.cubeSize = cubeSize
        self.node_num = node_num
        self.gap = gap
        self.relative_posi = relative_posi

    def run(self):
        pygame.init()

        if self.is2D:
            screen = Screen(self.screensize[0],self.screensize[1],scale=1.5)
        else:
            screen = PerspectiveScreen(self.screensize[0],self.screensize[1],scale=1.5)

        cube = Cube(self.cubeSize[0],self.cubeSize[1],self.cubeSize[2])
        grid = Grid(self.node_num, self.gap)

        if self.control_input == 'test':
            q = Quaternion(1,0,0,0)
            incr = Quaternion(0.96,0.01,0.01,0).normalized()
            count = 0

            while 1:
                q = q * incr * incr * incr
                if self.isGrid:
                    grid.draw(screen)
                cube.draw(screen,q,Vector3(count,-count,count))       
                event = pygame.event.poll()
                if event.type == pygame.QUIT \
                    or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    break
                pygame.display.flip()
                pygame.time.delay(100) 
                cube.erase(screen)
                count+=1

        elif self.control_input == 'ue':
            # while True:
            #     q = 
            #     posi =   #Vector3
               
            #     if isGrid:
            #         grid.draw(screen)
            #     cube.draw(screen,q,posi)       
            #     event = pygame.event.poll()
            #     if event.type == pygame.QUIT \
            #         or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            #         break
            #     pygame.display.flip()
            #     pygame.time.delay(100) 
            #     cube.erase(screen)

            pass
        

        elif self.control_input == 'imu':
            imuc = IMUTCPClient.IMUTCPClient()
            while True:
                control = imuc.recv_imu_update()
                if control is not None:
                    q = Quaternion(control["quatW"], control["quatX"], control["quatY"], control["quatZ"]).normalized()
                    posi = Vector3(0,0,0)
               
                    if self.isGrid:
                        grid.draw(screen)
                    cube.draw(screen,q,posi)       
                    event = pygame.event.poll()
                    if event.type == pygame.QUIT \
                        or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        break
                    pygame.display.flip()
                    pygame.time.delay(100) 
                    cube.erase(screen)
            

        else:
            print('invalid control input!')
      

if __name__ == "__main__":
    gui = PygameImuGui(node_num = 10, gap = 10)
    gui.run()
