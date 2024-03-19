# name: imu_const.py
# description: Constants for IMU data
# author: Vu Phan
# date: 2023/06/06


""" Selected filters for comparison (to be continued)
	MAH: Mahony filter
	MAD: Madgwick filter
	COMP: Vanilla complementary filter
	EKF: Extended Kalman filter
	Xsens: built-in filter from Xsens (already provided by Xsens)
	SAAM: Super-fast attitude from accelerometer and magnetometer
	RIANN: Robust IMU-based attitude neural network
	VQF: Versatile quaternion-based filter
"""
AVAILABLE_FILTERS = ['MAH', 'MAD', 'COMP', 'EKF', 'Xsens', 'SAAM', 'RIANN', 'VQF']


""" Sensor ID
"""
SENSOR = {'CHEST': 				'00B4D7D4',
			'PELVIS': 			'00B4D7D3', 
			'THIGH_L_MID': 		'00B4D7FD', 
			'SHANK_L_MID': 		'00B4D7CE', 
			'FOOT_L': 			'00B4D7FF', 
			'THIGH_R_MID': 		'00B4D6D1', 
			'SHANK_R_MID': 		'00B4D7FB', 
			'FOOT_R': 			'00B4D7FE', 
			'THIGH_R_HIGH': 	'00B4D7D0', 
			'THIGH_R_LOW': 		'00B4D7D8', 
			'THIGH_R_FRONT': 	'00B4D7D1', 
			'SHANK_R_HIGH': 	'00B4D7BA', 
			'SHANK_R_LOW': 		'00B4D7D5', 
			'SHANK_R_FRONT': 	'00B42961', 
			'THIGH_L_HIGH': 	'00B4D7D2', 
			'THIGH_L_LOW': 		'00B4D7CD', 
			'THIGH_L_FRONT': 	'00B4D7D6', 
			'SHANK_L_HIGH': 	'00B4D7CF', 
			'SHANK_L_LOW': 		'00B4D7FA', 
			'SHANK_L_FRONT': 	'00B42991'}



