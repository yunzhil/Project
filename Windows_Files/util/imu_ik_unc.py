# name: imu_ik_unc.py
# description: Perform (unconstrained) inverse kinematics for IMU data
# author: Vu Phan
# date: 2023/06/05


import numpy as np 
import pandas as pd
import quaternion
import math

import sys, os
sys.path.append('/path/to/IMU_Kinematics_Comparison')

from util.filters import *
from util.quaternion_handling import *


def get_orientation_imu(filter_type, dim, imu_dt, imu_placement_id, sampling_freq, init_orientation_4_6d = None, params = None):
	""" Get orientation (in quaternion) from IMU data

	Params:
		filter_type: type of sensor fusion algorithm to be applied | str
		dim: '6D' or '9D' filter | str
		imu_dt: raw data of IMU | dict of pd.DataFrame
		imu_placement_id: IDs of IMUs on each body segment | dict of str | check config.py
		sampling_freq: sampling frequency of IMU data | int
		init_orientation_4_6d: only applied for 6D filters | dict of np.array
		params: filter parameter(s), use default params if params = None | dict of int

	Returns:
		imu_orientation: orientation of each IMU (in quaternion) | dict of np.array
	"""
	imu_orientation = {'torso_imu': None, 'pelvis_imu': None,
						'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
						'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}

	if filter_type != 'Xsens':
		for segment in imu_placement_id.keys():
			gyr_data = imu_dt[segment][['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
			acc_data = imu_dt[segment][['Acc_X','Acc_Y','Acc_Z']].to_numpy()
			mag_data = imu_dt[segment][['Mag_X','Mag_Y','Mag_Z']].to_numpy()

			if filter_type == 'VQF':
				temp_estimation = vqf_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, params = params)
				imu_orientation[segment] = quaternion.as_quat_array(temp_estimation['quat' + dim])

				if dim == '6D':
					calib_earth = quaternion.as_quat_array(1*init_orientation_4_6d[segment])
					calib_earth = calib_earth[0]*imu_orientation[segment][0].conjugate()
					for i in range(len(imu_orientation[segment])):
						imu_orientation[segment][i] = calib_earth*imu_orientation[segment][i]

			elif filter_type == 'RIANN':
				temp_estimation = riann_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq)
				imu_orientation[segment] = quaternion.as_quat_array(temp_estimation)

				calib_earth = quaternion.as_quat_array(1*init_orientation_4_6d[segment])
				calib_earth = calib_earth[0]*imu_orientation[segment][0].conjugate()
				for i in range(len(imu_orientation[segment])):
					imu_orientation[segment][i] = calib_earth*imu_orientation[segment][i]

			else:
				if dim == '6D':
					q_init = init_orientation_4_6d[segment]
				else:
					q_init = None 

				if filter_type == 'MAH':
					temp_estimation = mah_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, q0 = q_init, params = params)

				elif filter_type == 'MAD':
					temp_estimation = mad_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, q0 = q_init, params = params)

				elif filter_type == 'COMP':
					temp_estimation = comp_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, q0 = q_init, params = params)

				elif filter_type == 'EKF':
					temp_estimation = ekf_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, q0 = q_init, params = params)

				elif filter_type == 'SAAM':
					temp_estimation = saam_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, params = params)
				
				elif filter_type == 'QUEST':
					temp_estimation = quest_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, params = params)

				elif filter_type == 'FLAE':
					temp_estimation = flae_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, params = params)

				elif filter_type == 'FQA':
					temp_estimation = fqa_filter(dim = dim, gyr = gyr_data, acc = acc_data, mag = mag_data, frequency = sampling_freq, params = params)

				else: 
					pass

				imu_orientation[segment] = quaternion.as_quat_array(temp_estimation.Q)

				if filter_type in ['SAAM', 'QUEST', 'FQA', 'FLAE']:
					calib_earth = quaternion.as_quat_array(1*init_orientation_4_6d[segment])
					calib_earth = calib_earth[0]*imu_orientation[segment][0].conjugate()
					for i in range(len(imu_orientation[segment])):
						imu_orientation[segment][i] = calib_earth*imu_orientation[segment][i]

	else:
		for segment in imu_placement_id.keys():
			imu_orientation[segment] = quaternion.as_quat_array(imu_dt[segment][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']].to_numpy())

	return imu_orientation

def sensor_to_body_calibration_imu(imu_static_orientation):
	""" Apply sensor to body calibration (more accurately, pelvis calibration)

	Params:
		imu_static_orientation: orientation of IMUs during the static trial | dict of (1x4) np.array

	Returns:
		cal_orientation: sensor-to-body calibration | dict of (1x4) np.array
	"""
	cal_orientation = {'cal_pelvis': None, 'cal_thigh_l': None, 'cal_thigh_r': None,
									'cal_shank_l': None, 'cal_shank_r': None, 
									'cal_foot_l': None, 'cal_foot_r': None} 

	cal_orientation['cal_pelvis']  = imu_static_orientation['pelvis_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_thigh_l'] = imu_static_orientation['femur_l_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_thigh_r'] = imu_static_orientation['femur_r_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_shank_l'] = imu_static_orientation['tibia_l_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_shank_r'] = imu_static_orientation['tibia_r_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_foot_l']  = imu_static_orientation['calcn_l_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 
	cal_orientation['cal_foot_r']  = imu_static_orientation['calcn_r_imu'].conjugate()*imu_static_orientation['pelvis_imu'] 

	return cal_orientation

def get_angle_bw_2_coords(q0_cal, q0, q1_cal, q1):
	""" Obtain angles between two coordinate frames

	Params:
		q0_cal: sensor-to-body calibration of q0 | (1x4) np.array
		q0: moving orientation of frame 0 | (Nx4) np.array
		q1_cal: sensor-to-body calibration of q1 | (1x4) np.array
		q1: moving orientation of frame 1 | (Nx4) np.array

	Returns:
		angles: moving angles between frame 0 and 1 | (Nx3) np.array
	"""
	angle_arr  = []
	num_sample = len(q0)

	for i in range(num_sample):
		q0_star 	= q0[i]*q0_cal
		q1_star 	= q1[i]*q1_cal
		q0_to_q1 	= quaternion.as_float_array(q0_star.conjugate()*q1_star)[0]
		angle 		= np.array(get_angle_from_quaternion(q0_to_q1))

		angle_arr.append(angle)

	angle_arr = np.array(angle_arr)

	return angle_arr

def get_all_ja_imu(cal_orientation, imu_orientation):
	""" Obtain joint angles from IMU data

	Params: 
		cal_orientation: sensor-to-body calibration obtained from imu_static_orientation | dict of (1x4) np.array
		imu_orientation: orientation of each IMU (in quaternion) | dict of (Nx4) np.array 

	Returns:
		imu_ja: IMU-based joint angles | dict of (Nx3) np.array
	"""

	imu_ja = {'hip_rotation_l': None, 'hip_flexion_l': None, 'hip_adduction_l': None, 'knee_angle_l': None, 'ankle_angle_l': None,
				'hip_rotation_r': None, 'hip_flexion_r': None, 'hip_adduction_r': None, 'knee_angle_r': None, 'ankle_angle_r': None}

	temp_hip_l = get_angle_bw_2_coords(cal_orientation['cal_pelvis'], imu_orientation['pelvis_imu'], cal_orientation['cal_thigh_l'], imu_orientation['femur_l_imu'])
	temp_knee_l = get_angle_bw_2_coords(cal_orientation['cal_thigh_l'], imu_orientation['femur_l_imu'], cal_orientation['cal_shank_l'], imu_orientation['tibia_l_imu'])
	temp_ankle_l = get_angle_bw_2_coords(cal_orientation['cal_shank_l'], imu_orientation['tibia_l_imu'], cal_orientation['cal_foot_l'], imu_orientation['calcn_l_imu'])
	imu_ja['hip_rotation_l']  = temp_hip_l[:, 0]
	imu_ja['hip_flexion_l']   = temp_hip_l[:, 1]
	imu_ja['hip_adduction_l'] = temp_hip_l[:, 2]
	imu_ja['knee_angle_l']    = temp_knee_l[:, 1]
	imu_ja['ankle_angle_l']   = temp_ankle_l[:, 1]

	temp_hip_r = get_angle_bw_2_coords(cal_orientation['cal_pelvis'], imu_orientation['pelvis_imu'], cal_orientation['cal_thigh_r'], imu_orientation['femur_r_imu'])
	temp_knee_r = get_angle_bw_2_coords(cal_orientation['cal_thigh_r'], imu_orientation['femur_r_imu'], cal_orientation['cal_shank_r'], imu_orientation['tibia_r_imu'])
	temp_ankle_r = get_angle_bw_2_coords(cal_orientation['cal_shank_r'], imu_orientation['tibia_r_imu'], cal_orientation['cal_foot_r'], imu_orientation['calcn_r_imu'])
	imu_ja['hip_rotation_r']  = temp_hip_r[:, 0]
	imu_ja['hip_flexion_r']   = temp_hip_r[:, 1]
	imu_ja['hip_adduction_r'] = temp_hip_r[:, 2]	
	imu_ja['knee_angle_r']    = temp_knee_r[:, 1]	
	imu_ja['ankle_angle_r']   = temp_ankle_r[:, 1]

	return imu_ja



