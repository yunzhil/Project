# name: preprocessing.py 
# description: Preprocess data 
# author: Vu Phan
# date: 2023/06/05


import pandas as pd 
import numpy as np 
import math

from tqdm import tqdm
from scipy import signal
from scipy.spatial.transform import Rotation as R


def get_imu_data(imu_fn_txt):
	""" Read and format data from a single Xsens IMU (.txt)

	Params:
		imu_fn_txt: filename of the IMU data | str

	Returns:
		imu_dt: formatted data of a single IMU | pd.DataFrame
	"""

	with open(imu_fn_txt, 'r') as f:
		txt     = f.readlines()
		header  = txt[4].split('\t')

	temp_dt     = np.genfromtxt(imu_fn_txt, delimiter='\t', skip_header=5)
	temp_dt     = temp_dt[:, 0::]
	header 		= header[0::]
	header[-1] 	= header[-1][0:-1]
	imu_dt 		= pd.DataFrame(temp_dt, columns = header)

	return imu_dt


def get_imu_data_xsens(imu_xsens_fn_xlsx, sheet):
	""" Read data from Xsens biomechanical model (.xlsx)

	Params:
		imu_xsens_fn_xlsx: filename of the Xsens biomechanical model results | str
		sheet: different types of data, e.g., acceleration, joint angles, etc. | str
	
	Returns:
		imu_xsens_dt: retrieved data from the Xsens biomechanical model Excel file | pd.DataFrame
	"""
	imu_xsens_dt = pd.read_excel(imu_xsens_fn_xlsx, sheet_name = sheet)

	return imu_xsens_dt


def get_avg_data(static_dt):
	""" Average mocap/IMU data of the static trial

	Params: 
		static_dt: mocap/IMU data during static trial | dict of (Nxm) pd.DataFrame

	Returns:
		avg_dt: Averaged mocap/IMU data during static trial | dict of (1xm) pd.DataFrame
	"""

	temp_dt = 1*static_dt.mean(axis = 0)
	avg_dt  = pd.DataFrame(temp_dt.transpose().values.reshape(1, -1), columns = temp_dt.index)

	return avg_dt


def match_imu_data(imu_dt, imu_placement_id):
	""" Match IMU data to avoid frame drops at the beginning and the end

	Params: 
		imu_dt: raw data of IMU | dict of pd.DataFrame
		imu_placement_id: IDs of IMUs on each body segment | dict of str | check config.py

	Returns:
		imu_dt_matched: matched IMU data | dict of pd.DataFrame
	"""
	count       = 1
	start_frame = 0
	stop_frame  = 0

	imu_dt_matched = {'torso_imu': None, 'pelvis_imu': None, 
					'calcn_r_imu': None, 'tibia_r_imu': None, 'femur_r_imu': None,
					'calcn_l_imu': None, 'tibia_l_imu': None, 'femur_l_imu': None}

	for segment in tqdm(imu_placement_id.keys()):
		if count == 1:
			temp_start = 10
			temp_stop   = np.where(imu_dt['pelvis_imu']['PacketCounter'] == imu_dt['pelvis_imu']['PacketCounter'].iloc[-100])[0][0]
			start_frame = imu_dt['pelvis_imu']['PacketCounter'][temp_start]
			stop_frame  = imu_dt['pelvis_imu']['PacketCounter'][temp_stop]
			count += 1

		else:
			temp_start = np.where(imu_dt[segment]['PacketCounter'] == start_frame)[0][0]
			temp_stop  = np.where(imu_dt[segment]['PacketCounter'] == stop_frame)[0][0]

		imu_dt_matched[segment] = imu_dt[segment].iloc[temp_start:temp_stop, :]
		print(imu_dt_matched[segment].shape)

	return imu_dt_matched


def get_optitrack_mocap_data_csv(mocap_fn_csv):
	""" Read and format mocap data collected from the OptiTrack system (.csv)

	Params: 
		mocap_fn_csv: filename of the mocap marker trajectories | str

	Returns:
		dt: formatted data of marker trajectories | pd.DataFrame
	"""
	
	dt = pd.read_csv(mocap_fn_csv, skiprows = 3, low_memory = False)
	dt = dt.iloc[:, 1:] 
	dt = dt.iloc[2:, :] 

	names_pos 	= list(dt.columns)
	names_pos	= [name.split(':')[1][0:4] for name in names_pos[1:]]
	names_pos	= [''] + names_pos
	names_axis	= dt.iloc[0, :]
	names 	 	= []
	for i in range(len(names_pos)):
		names.append(names_pos[i] + ' ' + names_axis[i])
	dt 			= dt.iloc[1:, :]
	dt.columns 	= names

	dt = dt.reset_index() 
	dt = dt.iloc[:, 1:] 

	dt = dt.astype('float64') 

	return dt


def get_optitrack_mocap_data_trc(mocap_fn_trc):
	""" Read and format mocap data collected from the OptiTrack system (.trc)

	Params: 
		mocap_fn_trc: filename of the mocap marker trajectories | str

	Returns:
		dt: formatted data of marker trajectories | pd.DataFrame
		time: time array of collecting marker trajectories | pd.DataFrame
	"""
	
	with open(mocap_fn_trc, 'r') as f:
		txt 	= f.readlines()
		header	= txt[3].split('\t')

	header 		= header[2::]
	d_header 	= []
	for i in range(len(header)):
		d_header.append(header[i].strip('\n') + ' X')
		d_header.append(header[i].strip('\n') + ' Y')
		d_header.append(header[i].strip('\n') + ' Z')

	value	= np.genfromtxt(mocap_fn_trc, delimiter='\t', skip_header=5)
	time	= 1*value[:, 1]
	value	= value[:, 2::]

	dt 		= pd.DataFrame(value, columns = d_header)
	time 	= pd.DataFrame(time, columns = ['Time'])

	return dt, time


def lowpass_filter(dt, dt_freq, cutoff_freq, filter_order):
	""" Low-pass filter the given data
	Params:
		dt: data to be filtered | pd.DataFrame
		dt_freq: original sampling frequency of the given data | int
		cutoff_freq: cut-off frequency of the filter | int
		filter_order: order of the filter | int

	Returns:
		filtered_dt: filtered data | pd.DataFrame
	"""

	Wn 						= cutoff_freq*2/dt_freq
	b, a 					= signal.butter(filter_order, Wn, btype = 'low')
	filtered_dt 			= signal.filtfilt(b, a, dt, axis = 0) 
	filtered_dt 			= pd.DataFrame(filtered_dt, columns = dt.columns)
	filtered_dt.iloc[:, 0] 	= dt.iloc[:, 0]

	return filtered_dt


def downsample(dt, dt_freq, targ_freq):
	""" Resample data (normally to match data from different modalities for comparison)

	Params:
		dt: data to be downsampled | pd.DataFrame
		dt_freq: original sampling frequency of the given data | int
		targ_freq: target frequency to be downsampled to | int

	Returns:
		ds_dt: downsampled data | pd.DataFrame
	"""

	num_curr_samples 	= dt.shape[0]
	num_targ_samples 	= int((num_curr_samples*targ_freq)/dt_freq)
	ds_dt 				= signal.resample(dt, num_targ_samples)

	ds_dt = pd.DataFrame(ds_dt, columns = dt.columns)

	return ds_dt


def downsample_matlab(dt, dt_freq, targ_freq, axis = 1):
	""" Decrease sample rate by integer factor (adopting MATLAB implementation of downsample)

	Params:
		dt: data to be downsampled | pd.DataFrame
		dt_freq: original sampling frequency of the given data | int
		targ_freq: target frequency to be downsampled to | int

	Returns:
		ds_dt: downsampled data | pd.DataFrame
	"""
	factor = int(dt_freq/targ_freq)

	if axis == 0:
		ds_dt = 1*dt.iloc[:, 0::factor]
	else:
		ds_dt = 1*dt.iloc[0::factor, :]

	print(ds_dt)
	
	return ds_dt







