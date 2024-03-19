# name: imu_ik_os.py
# description: Perform (constrained) inverse kinematics for IMU data using OpenSense (OpenSim)
# author: Vu Phan
# date: 2023/06/05


import opensim as osim
import numpy as np
import pandas as pd
import quaternion
import math


def convert_imu_orientation_to_os(subject, filter_type, imu_orientation, imu_placement_id, sampling_freq, stat_flag):
	""" Convert IMU orientation to the OpenSim format (i.e., .sto)

	Params:
		subject: subject ID | int
		filter_type: type of sensor fusion algorithm to be applied | str
		imu_orientation: orientation of each IMU (in quaternion) | dict of np.array
		imu_placement_id: IDs of IMUs on each body segment | dict of str | check config.py
		sampling_freq: sampling frequency of IMU data | int
		stat_flag: is this static trial (for calibration) or not | bool

	Returns:
		No return, but exporting an .sto file
	"""
	time = 0
	format_dt = 'DataRate=' + str(sampling_freq) + '\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4-2022-07-23-0e9fedc\nendheader\n'
	format_dt = format_dt + 'time'

	for segment in imu_orientation.keys():
		format_dt = format_dt + '\t' 
		format_dt = format_dt + segment
	format_dt = format_dt + '\n'

	num_samples = imu_orientation['pelvis_imu'].shape[0]
	dt          = 1.0/sampling_freq
	for i in range(num_samples):
		format_dt = format_dt + str(i*dt)

		for segment in imu_orientation.keys():
			format_dt = format_dt + '\t'
			temp_quat = quaternion.as_float_array(imu_orientation[segment][i])
			# temp_quat = imu_orientation[segment][i]
			# print(temp_quat)
			format_dt = format_dt + str(temp_quat[0]) + ',' + \
									str(temp_quat[1]) + ',' + \
									str(temp_quat[2]) + ',' + \
									str(temp_quat[3])

		format_dt = format_dt + '\n'

	if stat_flag == True:
		orientation_fn = 's' + str(subject) + '_cal_' + filter_type + '_orientation.sto'
	else:
		orientation_fn = 's' + str(subject) + '_' + filter_type + '_orientation.sto'
	with open('os_ik\\' + orientation_fn, 'w') as f:
		f.write(format_dt)

def os_calibration(orientation_cal_fn, visulizeCalibration):
	""" Perform calibration for the OpenSim model with the given orientation in the .sto file

	Params:
		orientation_cal_fn: filename of the .sto file containing orientation | str
		visulizeCalibration: to visualize the calibration or not | bool

	Returns:
		No return, but export a calibrated model 
	"""
	modelFileName               = 'Rajagopal_2015.osim'
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

def os_ik(orientationsFileName, visualizeTracking):
	""" Perform IK with OpenSim/OpenSense

	Params:
		orientationsFileName: filename of the .sto file containing orientation | str
		visualizeTracking: to visualize the calibration or not | bool

	Returns:
		No return, but export 
	"""
	startTime                   = 0
	endTime                     = 999
	modelFileName               = 'Rajagopal_2015.osim'
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

def get_all_ja_os(ik_fn):
	""" Get all joint angles from OpenSim/OpenSense

	Params:
		ik_fn: filename of the IK results from OpenSim | str

	Returns:
		imu_os_ja: OpenSim IMU-based joint angles | dict of np.array
	"""
	imu_os_ja = {'hip_rotation_l': None, 'hip_flexion_l': None, 'hip_adduction_l': None, 'knee_angle_l': None, 'ankle_angle_l': None,
				'hip_rotation_r': None, 'hip_flexion_r': None, 'hip_adduction_r': None, 'knee_angle_r': None, 'ankle_angle_r': None}

	with open('os_ik\\' + ik_fn, 'r') as f:
		txt    = f.readlines()
		header = txt[6].split('\t')

	angles = np.genfromtxt('os_ik\\' + ik_fn, delimiter='\t', skip_header=7)
	dt     = pd.DataFrame(angles, columns = header)

	imu_os_ja['hip_rotation_l']  = dt['hip_rotation_l']
	imu_os_ja['hip_flexion_l']   = dt['hip_flexion_l']
	imu_os_ja['hip_adduction_l'] = dt['hip_adduction_l']
	imu_os_ja['knee_angle_l']    = dt['knee_angle_l']
	imu_os_ja['ankle_angle_l']   = dt['ankle_angle_l']

	imu_os_ja['hip_rotation_r']  = dt['hip_rotation_r']
	imu_os_ja['hip_flexion_r']   = dt['hip_flexion_r']
	imu_os_ja['hip_adduction_r'] = dt['hip_adduction_r']
	imu_os_ja['knee_angle_r']    = dt['knee_angle_r']	
	imu_os_ja['ankle_angle_r']   = dt['ankle_angle_r']

	return imu_os_ja






