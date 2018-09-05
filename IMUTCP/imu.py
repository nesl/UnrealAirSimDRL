import serial

with serial.Serial('/dev/ttyACM0', 115200) as ser:
    line = ser.readline()
    #ser.write(b'ea')
    while True:
        line = ser.readline()
        data = [float(x) for x in str(line.decode('utf-8')).split(',')]
        print(data)
