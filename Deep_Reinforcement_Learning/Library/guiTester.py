# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 01:53:54 2018

@author: natsn
"""
import numpy as np
import multiprocessing
import AirSimGUI
import time

def test_inertial_display(pipe, vehicle_names):
    cur = np.array(np.random.rand(18))
    current_inertial_state = dict.fromkeys(vehicle_names, cur)
    
    print("Incrementing State Vector")
    while True:
        print("Incrementing State Vector")
        pipe.send(current_inertial_state)
        time.sleep(.25)
        cur += np.array(np.random.rand(18))
        current_inertial_state = dict.fromkeys(vehicle_names, cur)
        
def test_inertial_and_video_displays(pipeInert, pipeVideo, vehicle_names):
    cur = np.array(np.random.rand(18))
    current_inertial_state = dict.fromkeys(vehicle_names, cur)
    count_top = 100
    tensors_rgb = [dict.fromkeys(vehicle_names, np.array(255*np.random.rand(256,256,9), dtype = np.uint8)) for i in range(count_top)]
    print("Incrementing State Vector")
    count = 0
    while True:
        print("Incrementing State Vector")
        pipeInert.send(current_inertial_state)
        time.sleep(.05)
        pipeVideo.send(tensors_rgb[count])
        time.sleep(.05)
        cur += np.array(np.random.rand(18))
        current_inertial_state = dict.fromkeys(vehicle_names, cur)
        count += 1
        if count >= count_top:
            count = 0

if __name__ == '__main__':
    parStatePipe, chStatePipe = multiprocessing.Pipe()
    parVidPipe, chVidPipe = multiprocessing.Pipe()
    num_feeds = 3
    vehicle_names = ["Vehicle 1", "Vehicle 2"]
    app = AirSimGUI.QuadcopterGUI(parStatePipe, parVidPipe, vehicle_names, num_feeds, True)
    test_inertial_and_video_displays(chStatePipe, chVidPipe, vehicle_names)
    
