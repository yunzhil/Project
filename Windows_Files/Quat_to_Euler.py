import math
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
import numpy as np
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians\

file = open(r'C:\Users\kiddb\Downloads\eval_hip_adduction_001-000_00B4D6D1.txt', 'r')
data = file.readlines()
IMU_f = 'euler'

if IMU_f == 'euler':
        full_data = []
        for index, line in enumerate(data):
                if index > 4:
                        quat = list(map(float, line.rstrip('\n').split('\t')[-4:]))
                        euler_ang = euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])
                        full_data.append(euler_ang)

elif IMU_f == 'accel':
        full_data = []
        for index, line in enumerate(data):
                if index > 4:
                        accel = list(map(float, line.rstrip('\n').split('\t')[2:5]))
                        full_data.append(accel)

IMU_df = pd.DataFrame(full_data)
IMU_df.plot()
plt.show()
