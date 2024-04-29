# name: ik_mt.py
# description: perform IK to obtain joint kinematics from IMU data
# author: Vu Phan
# date: 2024/01/27


import numpy as np
import quaternion
import math

from tqdm import tqdm

import sys, os 
sys.path.append('/path/to/IMU_Kinematics_Comparison_v2')

from utils import constant_common
from utils.mt import constant_mt
from utils.mt import sfa


# Get orientation from all sensors
# NOTE: Ignore the pelvis in the current version
def get_imu_orientation_mt(imu_data_mt, f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = '9D', params = None):
    ''' Get orientation from all sensors

    Args:
        + imu_data_mt (dict of pd.DataFrame): data from all sensors
        + f_type (str): type of filter, e.g., VQF, EKF, etc.
        + fs (float): sampling rate
        + dim (str): dimension, 6D or 9D

    Returns:
        + imu_orientation_mt (dict of quaternion): orientation of all sensors
    '''
    imu_orientation_mt = {}
    print('fs = %s' %(fs))

    for sensor_name in tqdm(imu_data_mt.keys()):
        if f_type == 'Xsens':
            imu_orientation_mt[sensor_name] = quaternion.as_quat_array(imu_data_mt[sensor_name][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']].to_numpy())
        else:
            gyr = imu_data_mt[sensor_name][['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
            acc = imu_data_mt[sensor_name][['Acc_X','Acc_Y','Acc_Z']].to_numpy()
            if dim == '9D':
                mag = imu_data_mt[sensor_name][['Mag_X','Mag_Y','Mag_Z']].to_numpy()
            else:
                mag = None

            if f_type == 'VQF':
                temp_estimation = sfa.apply_vqf(gyr, acc, mag, dim, fs, params)
            elif f_type == 'MAD':
                temp_estimation = sfa.apply_madgwick(gyr, acc, mag, dim, fs, params)
            elif f_type == 'MAH':
                temp_estimation = sfa.apply_mahony(gyr, acc, mag, dim, fs, params)
            elif f_type == 'EKF':
                temp_estimation = sfa.apply_ekf(gyr, acc, mag, dim, fs, params)
            elif f_type == 'RIANN':
                temp_estimation = sfa.apply_riann(gyr, acc, fs)

            if f_type == 'VQF':
                imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation['quat' + dim])
            elif f_type == 'RIANN':
                imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation)
            else:
                imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation.Q)

    return imu_orientation_mt


# --- Convert quaternion to Euler angles --- #
# Source: MTw_Awinda_User_Manual.pdf (page 77)
def quat_to_euler(quat):
    ''' Convert a quaternion to Euler angles (Xsens sensor)

    Args:
        + quat (np.array): quaternion

    Returns:
        + x_angle, y_angle, z_angle (np.array): Euler angles
    '''
    x_angle = np.rad2deg(math.atan2(2*quat[2]*quat[3] + 2*quat[0]*quat[1],
                                    2*quat[0]**2 + 2*quat[3]**2 - 1))
    y_angle = np.rad2deg(math.asin(2*quat[1]*quat[3] - 2*quat[0]*quat[2]))
    z_angle = np.rad2deg(math.atan2(2*quat[1]*quat[2] + 2*quat[0]*quat[3],
                                    2*quat[0]**2 + 2*quat[1]**2 - 1))

    angles_3d = [x_angle, y_angle, z_angle]

    return angles_3d


# --- Get joint angles between two adjacent segments --- #
def get_ja(sframe_1, sframe_2, s2s_1, s2s_2, c_flag = True):
    ''' Get joint angles from the provided orientation

    Args:
        + sframe_1, sframe_2 (quat_array, len = N): orientation of two adjacent sensors
        + s2s_1, s2s_2 (np.array, shape = 3x3): segment-to-sensor 1 and 2
        + c_flag (bool): enable (True) or disable (False) calibration

    Returns:
        + imu_ja (np.array, shape = N x 3): 3-dof angles of a joint
    '''

    N = sframe_1.shape[0]
    imu_ja = []

    s2s_1 = quaternion.from_rotation_matrix(s2s_1)
    s2s_2 = quaternion.from_rotation_matrix(s2s_2)

    if c_flag:
        segment_1 = [sframe_1[i]*s2s_1.conjugate() for i in range(N)]
        segment_2 = [sframe_2[i]*s2s_2.conjugate() for i in range(N)]
    else:
        segment_1 = 1*sframe_1
        segment_2 = 1*sframe_2
    
    joint_rot = [segment_1[i].conjugate()*segment_2[i] for i in range(N)]
    joint_rot = quaternion.as_float_array(joint_rot)
    imu_ja    = [quat_to_euler(joint) for joint in joint_rot]
    imu_ja    = np.array(imu_ja)

    # for i in tqdm(range(N)):
    #     if c_flag:
    #         segment_1 = sframe_1[i] * s2s_1.conjugate()
    #         segment_2 = sframe_2[i] * s2s_2.conjugate()
    #         joint_rot = segment_1.conjugate() * segment_2
    #     else:
    #         joint_rot = sframe_1[i].conjugate() * sframe_2[i]

    #     joint_rot = quaternion.as_float_array(joint_rot)
    #     angles_3d = quat_to_euler(joint_rot)
    #     imu_ja.append(angles_3d)

    # imu_ja = np.array(imu_ja)
    assert imu_ja.shape == (N, 3), 'Incorrect data shape'

    return imu_ja


# --- Get 5-DOF lower limb joint angles --- #
def get_all_ja_mt(seg2sens, orientation_mt):
    ''' Obtain all joint angles from IMUs 

    Args:
        + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
        + orientation_mt (dict of quaternion): orientation of all sensors
    
    Returns:
        + mt_ja (dict of np.array): joit angles
    '''
    mt_ja = {}
    
    temp_hip_l   = get_ja(orientation_mt['pelvis'], orientation_mt['thigh_l'], seg2sens['pelvis'], seg2sens['thigh_l'], c_flag = True)
    temp_knee_l  = get_ja(orientation_mt['thigh_l'], orientation_mt['shank_l'], seg2sens['thigh_l'], seg2sens['shank_l'], c_flag = True)
    temp_ankle_l = get_ja(orientation_mt['shank_l'], orientation_mt['foot_l'], seg2sens['shank_l'], seg2sens['foot_l'], c_flag = True)

    mt_ja['hip_adduction_l'] = constant_common.JA_SIGN['hip_adduction_l']*temp_hip_l[:, 0]
    mt_ja['hip_rotation_l']  = -1*constant_common.JA_SIGN['hip_rotation_l']*temp_hip_l[:, 1]
    mt_ja['hip_flexion_l']   = constant_common.JA_SIGN['hip_flexion_l']*temp_hip_l[:, 2]
    mt_ja['knee_adduction_l'] = -1*constant_common.JA_SIGN['knee_adduction_l']*temp_knee_l[:, 0]
    mt_ja['knee_rotation_l']  = -1*constant_common.JA_SIGN['knee_rotation_l']*temp_knee_l[:, 1]
    mt_ja['knee_flexion_l']  = constant_common.JA_SIGN['knee_flexion_l']*temp_knee_l[:, 2]
    mt_ja['ankle_adduction_l']    = constant_common.JA_SIGN['ankle_angle_l']*temp_ankle_l[:, 0]
    mt_ja['ankle_rotation_l']    = -1*constant_common.JA_SIGN['ankle_angle_l']*temp_ankle_l[:, 1]
    mt_ja['ankle_flexion_l']    = constant_common.JA_SIGN['ankle_angle_l']*temp_ankle_l[:, 2]

    temp_hip_r   = get_ja(orientation_mt['pelvis'], orientation_mt['thigh_r'], seg2sens['pelvis'], seg2sens['thigh_r'], c_flag = True)
    temp_knee_r  = get_ja(orientation_mt['thigh_r'], orientation_mt['shank_r'], seg2sens['thigh_r'], seg2sens['shank_r'], c_flag = True)
    temp_ankle_r = get_ja(orientation_mt['shank_r'], orientation_mt['foot_r'], seg2sens['shank_r'], seg2sens['foot_r'], c_flag = True)

    mt_ja['hip_adduction_r'] = constant_common.JA_SIGN['hip_adduction_r']*temp_hip_r[:, 0]
    mt_ja['hip_rotation_r']  = -1*constant_common.JA_SIGN['hip_rotation_r']*temp_hip_r[:, 1]
    mt_ja['hip_flexion_r']   = constant_common.JA_SIGN['hip_flexion_r']*temp_hip_r[:, 2]
    mt_ja['knee_adduction_r'] = -1*constant_common.JA_SIGN['knee_adduction_r']*temp_knee_r[:, 0]
    mt_ja['knee_rotation_r']  = -1*constant_common.JA_SIGN['knee_rotation_r']*temp_knee_r[:, 1]
    mt_ja['knee_flexion_r']  = constant_common.JA_SIGN['knee_flexion_r']*temp_knee_r[:, 2]
    mt_ja['ankle_adduction_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 0]
    mt_ja['ankle_rotation_r']    = -1*constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 1]
    mt_ja['ankle_flexion_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 2]


    # mt_ja['hip_adduction_r'] = constant_common.JA_SIGN['hip_adduction_r']*temp_hip_r[:, 0]
    # mt_ja['hip_rotation_r']  = constant_common.JA_SIGN['hip_rotation_r']*temp_hip_r[:, 1]
    # mt_ja['hip_flexion_r']   = constant_common.JA_SIGN['hip_flexion_r']*temp_hip_r[:, 2]
    # mt_ja['knee_adduction_r'] = constant_common.JA_SIGN['knee_adduction_r']*temp_knee_r[:, 0]
    # mt_ja['knee_rotation_r']  = constant_common.JA_SIGN['knee_rotation_r']*temp_knee_r[:, 1]
    # mt_ja['knee_flexion_r']  = constant_common.JA_SIGN['knee_flexion_r']*temp_knee_r[:, 2]
    # mt_ja['ankle_adduction_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 0]
    # mt_ja['ankle_rotation_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 1]
    # mt_ja['ankle_flexion_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 2]

    return mt_ja



