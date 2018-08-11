from dronekit import connect, Command, LocationGlobal, Vehicle, VehicleMode
from pymavlink import mavutil
from my_vehicle import MyVehicle
import time, sys, argparse, math

def connecting():
    print("Connecting")
    connection_string = '127.0.0.1:14550'
    vehicle = connect(connection_string, wait_ready=True)#, vehicle_class=MyVehicle)
    return vehicle

def arm_and_takeoff(vehicle, altitude):
    print("Pre-arm checks")
    # while not vehicle.is_armable:
    #     print("Waiting for vehicle to initialise..")
    #     time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off")
    vehicle.simple_takeoff(altitude)

    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

if __name__ == "__main__":
    print("Let us take off")
    vehicle = connecting()
    arm_and_takeoff(vehicle, 2.5)
