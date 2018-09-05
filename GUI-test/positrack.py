import tkinter as tk
from tkinter.ttk import Label, Frame, Entry, Notebook, Combobox

import threading 
import multiprocessing
from mpl_toolkits.mplot3d import  axes3d,Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import time
import math

root = tk.Tk()
root.title("test gui")
nb = Notebook(root)


#Add the test tab
testframe = Frame(nb)
nb.add(testframe, text="for test")

#Grid the notebook
nb.grid(row = 0, column = 0)

#Configure the test tab
fig = plt.figure(1, figsize= (10,10))
ax = Axes3D(fig)

canvas = FigureCanvasTkAgg(fig,testframe)
canvas.get_tk_widget().grid(row = 1, column = 0)

last_pos = ['init','init','init']
#Start consequetive plotting


#PUT THE LOOP IN A UPDATE FUNCTION
for i in range(400):
    #pos = [np.random.rand(1),np.random.rand(1),np.random.rand(1)]
    pos = [i, math.sin(i), 0]  
    #pos = [i*math.cos(60+i*360*5), i*math.sin(60+i*360*5), -50*i]
    if last_pos[0]!='init':
        ax.plot3D([last_pos[0],pos[0]], [last_pos[1],pos[1]], zs=[last_pos[2],pos[2]], color='#91A8d0')
    ax.scatter(pos[0],pos[1],pos[2],c='#f7cac9')
    canvas.draw()
    
    last_pos = pos[:]
    
   
    
    time.sleep(0.01)

#click the close button on the gui to shut down
root.mainloop()



#comments
#-automatically change the axes limits
#each update: canvas.draw() one time