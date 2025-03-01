import FPA_algorithm
import gaitphase
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from utils.mt import constant_mt, preprocessing_mt, calibration_mt, ik_mt
import ipdb

def Rotr(r):
    return np.array([1, 0, 0], [0, m.cos(r), -m.sin(r)], [0,m.sin(r), m.cos(r)])

def Rotp(p):
    return np.array([m.cos(p), 0, m.sin(p)], [0, 1, 0], [-m.sin(p), 0, m.cos(p)])

def Roty(y):
    return np.array([m.cos(y), -m.sin(y), 0], [m.sin(y), m.cos(y), 0], [0, 0, 1])

def J(sr, sp, cr, cp):
    return np.array([0, -cp], [cr*cp, -sr*sp], [-sr*cp, -cr*sp])

def f(sr, sp, cr, cp, a):
    return np.array([-sp-a.iloc[0]/np.linalg.norm(a)], [sr*cp-a.iloc[1]/np.linalg.norm(a)], [cr*cp-a.iloc[2]/np.linalg.norm(a)])

def data_proc(directory, file):
    rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    #  Import data
    IMUdata = preprocessing_mt.load_data_mt(directory + '\\' + file)
    IMUdata[['Acc_X', 'Acc_Y', 'Acc_Z']] = IMUdata.apply(lambda x: pd.Series(np.matmul(rot_mat, np.array([x.Acc_X, x.Acc_Y, x.Acc_Z]))), axis=1)
    IMUdata[['Gyr_X', 'Gyr_Y', 'Gyr_Z']] = IMUdata.apply(lambda x: pd.Series(np.matmul(rot_mat, np.array([x.Gyr_X, x.Gyr_Y, x.Gyr_Z]))), axis=1)
    #for ind, i in enumerate(IMUdata.iterrows()):
        #IMUdata.iloc[ind][['Acc_X', 'Acc_Y', 'Acc_Z']] = np.matmul(rot_mat, np.array(IMUdata.iloc[ind][['Acc_X', 'Acc_Y', 'Acc_Z']]))
        #IMUdata.iloc[ind][['Gyr_X', 'Gyr_Y', 'Gyr_Z']] = np.matmul(rot_mat, np.array(IMUdata.iloc[ind][['Gyr_X', 'Gyr_Y', 'Gyr_Z']]))
    IMUdata.rename(columns={'Acc_X':'AccelX', 'Acc_Y':'AccelY', 'Acc_Z':'AccelZ', 'Gyr_X': 'GyroX', 'Gyr_Y': 'GyroY', 'Gyr_Z': 'GyroZ'}, inplace=True)

    return IMUdata

def fpa_calc():
    fpa_array = np.array([])
    gait = gaitphase.GaitPhase(datarate=60)
    fpa = FPA_algorithm.FPA(is_right_foot=True, datarate=60)
    for i in w.iterrows():
        ind = i[0]
        w.iloc[ind] = i = np.rad2deg(i[1])
        gait.update_gaitphase(i)
        fpa.update_FPA(IMUdata.iloc[ind], gait.gaitphase_old, gait.gaitphase)
        fpa_array = np.append(fpa_array, fpa.FPA_this_step)
        #print(fpa.FPA_this_step, fpa.FPA_last_step)
    fpa_array
    avg_fpa = np.mean(fpa_array)
    ordered_unique_fpa, index = np.unique(fpa_array, return_index=True)
    unique_fpa = fpa_array[np.sort(index)]
    print(unique_fpa)
    return fpa_array, unique_fpa, avg_fpa

if __name__ == '__main__':
    # User Variables
    directory = r'C:\Users\kiddb\Downloads'
    file = 'eval_overground_walking_001-000_00B4D7FF.txt'

    IMUdata = data_proc(directory, file)
    a = IMUdata[['AccelX', 'AccelY', 'AccelZ']]
    ax, ay, az = a['AccelX'], a['AccelY'], a['AccelZ']
    anorm = [ax, ay, az]/np.linalg.norm(a)
    g = np.array([0, 0, 1])
    w = IMUdata[['GyroX', 'GyroY', 'GyroZ']]
    #w.columns = ['GyroX', 'GyroY', 'GyroZ']
    wx, wy, wz = w['GyroX'], w['GyroY'], w['GyroZ']

    fpa_total, unique, avg = fpa_calc()
    plt.plot(fpa_total)
    plt.show()
    print(avg)

    racc = np.arctan(ay/az) #should be for individual
    pacc = np.arctan(ax/np.sqrt(ay**2 + az**2)) #should be for individual

#  Orientation Estimation (Gradient descent)
# Trajectory by strapdown integration
# stance phase identification (zero velocity detection)
# Roll/Pitch Integration (omega)
# Detect foot stationary phase, heel-strike, toe off
#self.my_FPA.update_FPA(data[SENSOR_FOOT], self.my_GP.gaitphase_old, self.my_GP.gaitphase)
# for i = 1:N
#   if i in the middle of nth step stance phase, initialize racc,pacc of i from accel
#       for j=i to toe off from n-1 step, calculate rgyr, pgyr of j from gyro
#           if j == foot stationary phase, apply correction of j from accel
#           Transform foot accel
#       calculate FPA
# Questions: 
# wmax = max gyro measurement error of each axis? 
# Is middle stance == foot stationary middle phase?
# How does update_gait function work? No documentation on output.
# Confused about r(j+1) it says work backwards, so is j+1 farther back in time? or forwards?
# Figure dout the size differences issue (used wrong atan) but now, I am confused why we need to use atan of specific points rather than of the entire array?
# Wouldn't it be easier to just do atan of entire array and use logical array to extract desired elements?
# Can I assume all norms to be the same kind of norm? forbenius? or L2? 
# Roll/Pitch Correction (alpha)
# Foot Stationary Phase Detection (omega, alpha)
#  Acceleration Transformation
#  FPA Estimation
