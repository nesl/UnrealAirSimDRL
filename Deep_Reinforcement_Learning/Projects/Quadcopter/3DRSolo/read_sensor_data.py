from dronekit import connect, Command, LocationGlobal, Vehicle, VehicleMode
from pymavlink import mavutil
from my_vehicle import MyVehicle
import time, sys, argparse, math
from subprocess import Popen

MAV_MODE_AUTO   = 4
# Note: this might need to be changed if necessary

class DroneStatus(object):
    def __init__(self):
        self.vehicle = self.connecting()

    def connecting(self):
        print("Connecting")
        connection_string = '127.0.0.1:14550'
        vehicle = connect(connection_string, wait_ready=True)
        return vehicle

    def raw_imu_callback(self, attr_name, value):
        print(value)

    def get_data(self):
        return "{0}".format(self.vehicle.attitude)
    def get_pwm(self):
        vehicle.channels.overrides = {'3':300}
        

    def get_velocity(self):
        return "{0}".format(self.vehicle.velocity)

    def read_data(self):
        # vehicle.add_attribute_listener('raw_imu', raw_imu_callback)
        # print('Display for 5 seconds')
        # time.sleep(5)
        # file = open('/tmp/test1.txt', 'w')
        # def wildcard_callback(self, attr_name, value):
        #     if "attitude" in attr_name or "imu" in attr_name or "IMU" in attr_name:
        #         # print(" CALLBACK: (%s): %s" % (attr_name,value))
        #         file.write("%s\n" % value)
        print("\nAdd attribute callback detecting ANY attribute change")
        # vehicle.add_attribute_listener('*', wildcard_callback)
        # time.sleep(1)
        # while True:
            # print('Version: {0}'.format(vehicle.version))
            # print('Type: {0}'.format(vehicle._vehicle_type))
            # print('Arm ed: {0}'.format(vehicle.armed))
            # print('System status: {0}'.format(vehicle.system_status.state))
            # print('GPS: {0}'.format(vehicle.gps_0))
            # print('Alt: {0}'.format(vehicle.location.global_relative_frame.alt))
            # print("\n Home location: %s" % vehicle.home_location)
            # time.sleep(1)
            # print("{0}".format(self.vehicle.attitude))
        # file.close()

    def get_location_offset_meters(self, original_location, dNorth, dEast, alt):
        """
        Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
        specified `original_location`. The returned Location adds the entered `alt` value to the altitude of the `original_location`.
        The function is useful when you want to move the vehicle around specifying locations relative to
        the current vehicle position.
        The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
        For more information see:
        http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
        """
        earth_radius=6378137.0 #Radius of "spherical" earth
        #Coordinate offsets in radians
        dLat = dNorth/earth_radius
        dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

        #New position in decimal degrees
        newlat = original_location.lat + (dLat * 180/math.pi)
        newlon = original_location.lon + (dLon * 180/math.pi)
        return LocationGlobal(newlat, newlon,original_location.alt+alt)

    def direction(self):
        print("\nChannel overrides: %s" % self.vehicle.channels.overrides)

        print("Set Ch2 override to 200 (indexing syntax)")
        self.vehicle.channels.overrides['2'] = 200
        print(" Channel overrides: %s" % self.vehicle.channels.overrides)
        print(" Ch2 override: %s" % self.vehicle.channels.overrides['2'])

        print("Set Ch3 override to 300 (dictionary syntax)")
        self.vehicle.channels.overrides = {'3':300}
        print(" Channel overrides: %s" % self.vehicle.channels.overrides)

        print("Set Ch1-Ch8 overrides to 110-810 respectively")
        self.vehicle.channels.overrides = {'1': 110, '2': 210,'3': 310,'4':4100, '5':510,'6':610,'7':710,'8':810}
        print(" Channel overrides: %s" % self.vehicle.channels.overrides)

    def PX4setMode(self, vehicle, mavMode):
        vehicle._master.mav.command_long_send(vehicle._master.target_system, vehicle._master.target_component,
                                                   mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                                   mavMode,
                                                   0, 0, 0, 0, 0, 0)

def direction(vehicle):
    print("\nChannel overrides: %s" % vehicle.channels.overrides)

    print("Set Ch2 override to 200 (indexing syntax)")
    vehicle.channels.overrides['1'] = 1010
    print(" Channel overrides: %s" % vehicle.channels.overrides)
    # print(" Ch2 override: %s" % vehicle.channels.overrides['2'])
    #
    # print("Set Ch3 override to 300 (dictionary syntax)")
    # vehicle.channels.overrides = {'3':300}
    # print(" Channel overrides: %s" % vehicle.channels.overrides)

    print("Set Ch1-Ch8 overrides to 110-810 respectively")
    # vehicle.channels.overrides = {'1': 110, '2': 210,'3': 310,'4':4100, '5':510,'6':610,'7':710,'8':810}
    print(" Channel overrides: %s" % vehicle.channels.overrides)

def main():
    connection_string = '127.0.0.1:14550'
    vehicle = connect(connection_string, wait_ready=True)
    vehicle.mode = VehicleMode("MANUAL")
    direction(vehicle)
    # reboot()
    # try:
    #     drone_status = DroneStatus()
    # except:
    #     reboot()
    # drone_status = DroneStatus()
    # drone_status.direction()
    # vehicle = drone_status.connecting()
    # print('Arm ed: {0}'.format(vehicle.armed))
    # vehicle.armed = True
    # # vehicle.simple_takeoff(10)
    # time.sleep(1)
    # print('Arm ed: {0}'.format(vehicle.armed))
    # drone_status.PX4setMode(vehicle, MAV_MODE_AUTO)
    # # time.sleep(1)
    # home = vehicle.location.global_relative_frame
    # wp = drone_status.get_location_offset_meters(home, 0, 0, 10);
    # print(drone_status.get_data())

if __name__ == '__main__':
    main()
