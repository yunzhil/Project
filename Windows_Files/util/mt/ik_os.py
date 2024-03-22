# name: ik_os.py
# description: Perform (constrained) inverse kinematics for IMU data using OpenSense (OpenSim)
# author: Vu Phan
# date: 2023/06/05


import opensim as osim
import numpy as np
import pandas as pd
import quaternion
import math

import sys, os 
sys.path.append('/path/to/IMU_Kinematics_Comparison_v2')

from util.mt import constant_mt


def convert_imu_orientation_to_os(subject, f_type, orientation_mt, fs, stat_flag):
	""" Convert IMU orientation to the OpenSim format (i.e., .sto)

	Params:
		subject (int): subject ID 
		f_type (str): type of sensor fusion algorithm to be applied | str
		orientation_mt: orientation of each IMU (in quaternion) | dict of np.array
		fs: sampling frequency of IMU data | int
		stat_flag: is this static trial (for calibration) or not | bool

	Returns:
		No return, but exporting an .sto file
	"""
	time = 0
	format_dt = 'DataRate=' + str(fs) + '\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4-2022-07-23-0e9fedc\nendheader\n'
	format_dt = format_dt + 'time'

	for segment in orientation_mt.keys():
		format_dt = format_dt + '\t' 
		format_dt = format_dt + constant_mt.MT_TO_OPENSENSE_MAP[segment]
	format_dt = format_dt + '\n'

	num_samples = orientation_mt['pelvis'].shape[0]
	dt          = 1.0/fs
	for i in range(num_samples):
		format_dt = format_dt + str(i*dt)

		for segment in orientation_mt.keys():
			format_dt = format_dt + '\t'
			temp_quat = quaternion.as_float_array(orientation_mt[segment][i])
			# temp_quat = orientation_mt[segment][i]
			# print(temp_quat)
			format_dt = format_dt + str(temp_quat[0]) + ',' + \
									str(temp_quat[1]) + ',' + \
									str(temp_quat[2]) + ',' + \
									str(temp_quat[3])

		format_dt = format_dt + '\n'

	if stat_flag == True:
		orientation_fn = 's' + str(subject) + '_cal_' + f_type + '_orientation.sto'
	else:
		orientation_fn = 's' + str(subject) + '_' + f_type + '_orientation.sto'
	with open('os_ik\\' + orientation_fn, 'w') as f:
		f.write(format_dt)

def os_calibration(orientation_cal_fn, os_model, visulizeCalibration):
	""" Perform calibration for the OpenSim model with the given orientation in the .sto file

	Params:
		orientation_cal_fn: filename of the .sto file containing orientation | str
		os_model (str): selected skeletal model
		visulizeCalibration: to visualize the calibration or not | bool

	Returns:
		No return, but export a calibrated model 
	"""
	modelFileName               = os_model + '.osim'
	orientationsFileName        = 'os_ik\\' + orientation_cal_fn
	sensor_to_opensim_rotations = osim.Vec3(-math.pi/2, 0, 0)
	baseIMUName                 = 'pelvis_imu'
	baseIMUHeading              = '-z'

	imuPlacer = osim.IMUPlacer()
	imuPlacer.set_model_file(modelFileName);
	imuPlacer.set_orientation_file_for_calibration(orientationsFileName)
	imuPlacer.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
	imuPlacer.set_base_imu_label(baseIMUName)
	imuPlacer.set_base_heading_axis(baseIMUHeading)
	imuPlacer.run(visulizeCalibration)

	model = imuPlacer.getCalibratedModel()
	model.printToXML('os_ik\\calibrated_' + modelFileName)

def os_ik(orientationsFileName, os_model, visualizeTracking):
	""" Perform IK with OpenSim/OpenSense

	Params:
		orientationsFileName: filename of the .sto file containing orientation | str
		os_model (str): selected skeletal model
		visualizeTracking: to visualize the calibration or not | bool

	Returns:
		No return, but export 
	"""
	startTime                   = 0
	endTime                     = 999
	modelFileName               = os_model + '.osim'
	sensor_to_opensim_rotations = osim.Vec3(-math.pi/2, 0, 0)
	resultsDirectory            = 'os_ik';

	imuIK = osim.IMUInverseKinematicsTool()
	imuIK.set_model_file('os_ik\\' + 'calibrated_' + modelFileName)
	imuIK.set_orientations_file('os_ik\\' + orientationsFileName)
	imuIK.set_sensor_to_opensim_rotations(sensor_to_opensim_rotations)
	imuIK.set_results_directory(resultsDirectory)
	imuIK.set_time_range(0, startTime)
	imuIK.set_time_range(1, endTime)
	imuIK.run(visualizeTracking)

def get_all_ja_os(ik_fn, os_model):
	""" Get all joint angles from OpenSim/OpenSense

	Params:
		ik_fn: filename of the IK results from OpenSim | str

	Returns:
		imu_os_ja: OpenSim IMU-based joint angles | dict of np.array
	"""
	imu_os_ja = {}

	with open('os_ik\\' + ik_fn, 'r') as f:
		txt    = f.readlines()
		header = txt[6].split('\t')

	angles = np.genfromtxt('os_ik\\' + ik_fn, delimiter='\t', skip_header=7)
	dt     = pd.DataFrame(angles, columns = header)

	imu_os_ja['hip_adduction_l'] = 1*dt['hip_adduction_l'].to_numpy()
	imu_os_ja['hip_rotation_l']  = 1*dt['hip_rotation_l'].to_numpy()
	imu_os_ja['hip_flexion_l']   = 1*dt['hip_flexion_l'].to_numpy()
	if os_model == 'gait2392_simbody':
		imu_os_ja['knee_flexion_l']  = -1*dt['knee_angle_l'].to_numpy()
	else:
		imu_os_ja['knee_flexion_l']  = 1*dt['knee_angle_l'].to_numpy()
	imu_os_ja['ankle_angle_l']   = 1*dt['ankle_angle_l'].to_numpy()

	imu_os_ja['hip_adduction_r'] = 1*dt['hip_adduction_r'].to_numpy()
	imu_os_ja['hip_rotation_r']  = 1*dt['hip_rotation_r'].to_numpy()
	imu_os_ja['hip_flexion_r']   = 1*dt['hip_flexion_r'].to_numpy()
	if os_model == 'gait2392_simbody':
		imu_os_ja['knee_flexion_r']  = -1*dt['knee_angle_r'].to_numpy()
	else:
		imu_os_ja['knee_flexion_r']  = 1*dt['knee_angle_r'].to_numpy()
	imu_os_ja['ankle_angle_r']   = 1*dt['ankle_angle_r'].to_numpy()

	return imu_os_ja






