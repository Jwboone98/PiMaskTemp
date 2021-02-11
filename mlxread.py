from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
print("Ambient Temperature :", sensor.get_ambient())
print("Object 1 Temperature :", sensor.get_object_1())
# print("Object 2 Temperature :", sensor.get_object_2())

def c_to_f(celc):
    return ((celc * 1.8) + 32)

x = 0

while x < 1000000000000000:
    if x % 50000 == 0:
        ambientF = c_to_f(sensor.get_ambient())
        objectF = c_to_f(sensor.get_object_1())
        print("Ambient Temperature :", ambientF)
        print("Object 1 Temperature :", objectF)
    x = x + 1
    
bus.close()
