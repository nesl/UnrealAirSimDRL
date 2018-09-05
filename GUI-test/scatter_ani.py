import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(0,10)
plt.ylim(0,10)

def animate(i):
    x.append(np.random.rand(1)*10)
    y.append(np.random.rand(1)*10)
    # for i in range(1000):
    #    x.append(0.1*i)
    #    y.append(0.1*i)
    sc.set_offsets(np.c_[x,y])

ani = matplotlib.animation.FuncAnimation(fig, animate, 
                frames=2, interval=100, repeat=True) 
plt.show()