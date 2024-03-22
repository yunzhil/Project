# name: common.py
# description: Contain common functions for use
# author: Vu Phan
# date: 2023/06/05


import math
import numpy as np 
from scipy.stats import pearsonr
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('/path/to/IMU_Kinematics_Comparison')
# Constants
from constants.common_const import *
from constants.imu_const import *
from constants.mocap_const import *


# --- Make folder if not existing --- #
def mkfolder(path):
	""" Create a folder in the given path if not exist
	"""
	if not os.path.exists(path):
		os.mkdir(path)


# --- IMU configurations --- #
def config_imu_placement(subject):
	""" Get IMU ID 

	Params:
		subject: subject ID | int

	Returns:
		imu_placement_id: IMU ID | dict
	"""
	if subject != 1:
		imu_placement_id = {'torso_imu': SENSOR['CHEST'], 'pelvis_imu': SENSOR['PELVIS'], 
							'calcn_r_imu': SENSOR['FOOT_R'], 'tibia_r_imu': SENSOR['SHANK_R_MID'], 'femur_r_imu': SENSOR['THIGH_R_MID'],
							'calcn_l_imu': SENSOR['FOOT_L'], 'tibia_l_imu': SENSOR['SHANK_L_MID'], 'femur_l_imu': SENSOR['THIGH_L_MID']} 
	else:
		imu_placement_id = {'torso_imu': SENSOR['CHEST'], 'pelvis_imu': SENSOR['PELVIS'], 
							'calcn_r_imu': SENSOR['FOOT_R'], 'tibia_r_imu': SENSOR['SHANK_R_MID'], 'femur_r_imu': SENSOR['THIGH_R_MID'],
							'calcn_l_imu': SENSOR['FOOT_L'], 'tibia_l_imu': SENSOR['THIGH_L_MID'], 'femur_l_imu': SENSOR['SHANK_L_MID']}
	
	return imu_placement_id


# --- Conversion to Euler angles --- #
# From quaternions of Xsens sensors
# Source: MTw_Awinda_User_Manual.pdf (page 77)
def quat_to_euler(quat):
    ''' Convert a quaternion to Euler angles (Xsens sensor)

    Args:
        + quat (np.array): quaternion

    Returns:
        + angle (np.array): Euler angles
    '''
    angle_x = np.rad2deg(math.atan2(2*quat[2]*quat[3] + 2*quat[0]*quat[1], 2*quat[0]**2 + 2*quat[3]**2 - 1))
    angle_y = np.rad2deg(math.asin(2*quat[1]*quat[3] - 2*quat[0]*quat[2]))
    angle_z = np.rad2deg(math.atan2(2*quat[1]*quat[2] + 2*quat[0]*quat[3], 2*quat[0]**2 + 2*quat[1]**2 - 1))

    angle = np.array([angle_x, angle_y, angle_z])

    return angle


# From rotation matrices
def rotmat_to_angle(rotmat):
	''' Convert a rotation matrix to Euler angles

	Args:
		+ t_mat (np.array): rotation matrix

	Returns:
		+ angle (np.array): Euler angles
	'''
	r     = R.from_matrix(rotmat)
	angle = r.as_euler('xyz', degrees = True)

	return angle


# --- Metrics for evaluation --- #
# RMSE
def get_rmse(mocap, imu):
	""" Compute root-mean-square error (RMSE) between mocap- and IMU-based joint angles

	Params:
		+ mocap (np.array): a joint angle computed using mocap 
		+ imu (np.array): a joint angle computed using IMU

	Returns:
		+ rmse (float): RMSE between mocap- and IMU-joint angles
	"""
	mse = np.nanmean(np.square(np.subtract(mocap, imu)))
	rmse = math.sqrt(mse)

	return rmse


# Maximum absolute error
def get_maxae(mocap, imu):
	""" Compute maximum absolute error (MaxAE) between mocap- and IMU-based joint angles

	Params:
		mocap (np.array): a joint angle computed using mocap 
		imu (np.array): a joint angle computed using IMU 

	Returns:
		mae (float): RMSE between mocap- and IMU-joint angles 
	"""
	mae = np.max(np.abs(np.subtract(mocap, imu)))

	return mae


# Pearson correlation coefficient
def get_corrcoef(mocap, imu):
	""" Compute correlation coefficient (r) between mocap- and IMU-based joint angles

	Params:
		mocap (np.array): a joint angle computed using mocap 
		imu (np.array): a joint angle computed using IMU 

	Returns:
		corr_coef (float): correlation coefficient between mocap- and IMU-based joint angles
	"""
	corr_coef, _ = pearsonr(mocap, imu)

	return corr_coef


