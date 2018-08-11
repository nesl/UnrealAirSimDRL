# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:30:46 2017

@author: natsn
"""

# Script to import the CSV Reaward and state data for plotting and analysis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

# 1) Import Data
data = pd.read_csv('RL_Data.csv')

round_num = data['Round'].values
action = data['Action'].values
posz = data['PosZ'].values
vz = data['Vz'].values
reward = data['Reward'].values
pitch = data['Pitch'].values 
roll = data['Roll'].values

# Intervention Reward signal meshplot
maxpitchroll = np.pi/3
posZ = np.arange(.005, 10, .005)
velZ = np.arange(-7,0, .005)
pitch_plus_act = np.arange(-np.pi/3 +.00001, np.pi/3 -.00001, .001)
roll_plus_act = np.arange(-np.pi/3 +.00001, np.pi/3 - .00001, .001)

posZ_r1, velZ_r1 = np.meshgrid(posZ, velZ)
posZ_r2, pitch_plus_act_r2 = np.meshgrid(posZ, pitch_plus_act)
posZ_r3, roll_plus_act_r3 = np.meshgrid(posZ, roll_plus_act)

# Backup Reward functions:
#r1 = np.tanh(posZ_r1/(.7*velZ_r1)) # better
#r1 = np.tanh(posZ_r1/(velZ_r1)) # good
#r1 = np.arctan(posZ_r1/(.5*velZ_r1)) # good
#r1 = -.33*posZ_r1**2/velZ_r1**2
#r2 = -posZ_r2 / np.sqrt((np.abs(np.log((maxpitchroll - np.abs(pitch_plus_act_r2))/maxpitchroll))))
#r3 = -posZ_r3 / np.sqrt((np.abs(np.log((maxpitchroll - np.abs(roll_plus_act_r3))/(maxpitchroll)))))

# Intervention reward signals
r1_i = 5*np.tanh(-.03*(posZ_r1+1.8)**1.9 / (np.abs(velZ_r1 + .001)**.7)) #better
r2_i = 5*np.tanh(-6*(posZ_r2+.3)**2*(maxpitchroll - np.abs(pitch_plus_act_r2))**6) # perfect
r3_i = 5*np.tanh(-6*(posZ_r3+.3)**2*(maxpitchroll - np.abs(roll_plus_act_r3))**6) # perfect

fig1 = plt.figure(1)
ax = fig1.gca(projection = '3d')
ax.set_title("Intervention Reward Signal 1")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Velocity (m/s)")
ax.plot_surface(posZ_r1, velZ_r1, r1_i, cmap = cm.coolwarm)

fig2 = plt.figure(2)
ax = fig2.gca(projection = '3d')
ax.set_title("Intervention Reward Signal 2")
ax.set_xlabel("Position (m)")
ax.set_ylabel("pitch (rad)")
ax.plot_surface(posZ_r2, pitch_plus_act_r2, r2_i, cmap = cm.coolwarm)

fig3 = plt.figure(3)
ax = fig3.gca(projection = '3d')
ax.set_title("Intervention Reward Signal 3")
ax.set_xlabel("Position (m)")
ax.set_ylabel("roll (rad)")
ax.plot_surface(posZ_r3, roll_plus_act_r3, r3_i, cmap = cm.coolwarm)


# Non-Intervention Reward signals:
#r1_ni = 5*np.tanh(-np.sqrt(np.abs(velZ_r1+ .001)/(3*posZ_r1**2))) #eh
#r1_ni = 2*GainNI*np.tanh(-np.sqrt(np.abs(1.9*velZ+ .001)**2/(3*(posZ+ .3)**2.5))) #good
r1_ni = 5*np.tanh(-np.abs(velZ_r1 + .15)**2/(.2*(posZ_r1+ .3)**3)) #better

#r2_ni = 5*np.tanh(1/(-.65*posZ_r2*(maxpitchroll - np.abs(pitch_plus_act_r2))**3)) # perfect
r2_ni = 5*np.tanh(1/(-8*(posZ_r2+.7)**2*(maxpitchroll - np.abs(pitch_plus_act_r2))**6)) # perfect
#r2_ni = 5*np.tanh(1/(-3*posZ_r2**2*(maxpitchroll - np.abs(pitch_plus_act_r2))**4)) # perfect
r3_ni = 5*np.tanh(1/(-8*(posZ_r3+.7)**2*(maxpitchroll - np.abs(roll_plus_act_r3))**6)) # perfect

fig4 = plt.figure(4)
ax = fig4.gca(projection = '3d')
ax.set_title("Non-Intervention Reward Signal 1")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Velocity (m/s)")
ax.plot_surface(posZ_r1, velZ_r1, r1_ni, cmap = cm.coolwarm)

fig5 = plt.figure(5)
ax = fig5.gca(projection = '3d')
ax.set_title("Non-Intervention Reward Signal 2")
ax.set_xlabel("Position (m)")
ax.set_ylabel("pitch (rad)")
ax.plot_surface(posZ_r2, pitch_plus_act_r2, r2_ni, cmap = cm.coolwarm)

fig6 = plt.figure(6)
ax = fig6.gca(projection = '3d')
ax.set_title("Non-Intervention Reward Signal 3")
ax.set_xlabel("Position (m)")
ax.set_ylabel("roll (rad)")
ax.plot_surface(posZ_r3, roll_plus_act_r3, r3_ni, cmap = cm.coolwarm)




#
# 2) Sort Data for data analysis
act0_data = {'Round':[], 'Action': [], 'PosZ': [], 'Vz': [], 'Reward': [], 'Pitch': [], 'Roll': []}
act1_data = {'Round':[], 'Action': [], 'PosZ': [], 'Vz': [], 'Reward': [], 'Pitch': [], 'Roll': []}
for pos,v,r,a,ro,pit,rol in zip(posz,vz,reward,action,round_num,pitch,roll):
    if a == 0:
        act0_data['Round'].append(ro)
        act0_data['Action'].append(a)
        act0_data['PosZ'].append(pos)
        act0_data['Vz'].append(v)
        act0_data['Reward'].append(r)
        act0_data['Pitch'].append(pit)
        act0_data['Roll'].append(rol)
    else:
        act1_data['Round'].append(ro)
        act1_data['Action'].append(a)
        act1_data['PosZ'].append(pos)
        act1_data['Vz'].append(v)
        act1_data['Reward'].append(r)
        act1_data['Pitch'].append(pit)
        act1_data['Roll'].append(rol)


# 1) Plot velocity vs reward for action 0 and action 1
plt.figure(2)
plt.subplot(2,1,1)
plt.title("Velocity vs Reward for Action 1")
plt.ylabel("Reward")
plt.xlabel("Velocity")

plt.subplot(2,1,2)
plt.title("Velocity vs Reward for Action 0")
plt.ylabel("Reward")
plt.xlabel("Velocity")


# 2) Plot velocity and position vs reward for action 0 and 1
fig3 = plt.figure(3)
ax = fig3.add_subplot(211, projection = '3d')
ax.set_title('Position, Velocity, Rewards for Action 0')
ax.scatter(act0_data['Vz'], act0_data['PosZ'],act0_data['Reward'])

ax = fig3.add_subplot(211, projection = '3d')
ax.set_title('Position, Velocity, Rewards for Action 1')
ax.scatter(act0_data['Vz'], act0_data['PosZ'],act0_data['Reward'])


# 3) Plot position and pitch vs reward for actions 0 and 1 


# 4) Plot position and roll vs reward for actions 0 and 1


# 5) Plot round by round 



























