# name: filters.py
# description: Sensor fusion methods for orientation estimation
# author: Vu Phan
# date: 2023/06/06


import numpy as np
from scipy import signal
from ahrs.filters import Mahony, Madgwick, EKF, Complementary, saam, quest, flae, fqa
from ahrs.util import wmm
from vqf import VQF, PyVQF
from riann.riann import RIANN

import sys, os
sys.path.append('/path/to/IMU_Kinematics_Comparison')

from util.one_euro_filter import OneEuroFilter


def vqf_filter(dim, gyr, acc, mag, frequency, params = None):
	""" Return 6D or 9D estimated orientation using VQF
	"""
	if params == None:
		vqf = PyVQF(gyrTs = 1.0/frequency)
	else:
		vqf = PyVQF(gyrTs = 1.0/frequency, tauAcc = params[0], tauMag = params[1])

	if dim == '6D':
		temp_estimation = vqf.updateBatch(gyr, acc)

	elif dim == '9D':
		temp_estimation = vqf.updateBatch(gyr, acc, mag)

	else:
		pass 

	return temp_estimation


def riann_filter(dim, gyr, acc, mag, frequency):
	""" Return 6D estimated orientation using RIANN
	"""
	riann = RIANN()

	if dim == '6D':
		temp_estimation = riann.predict(acc = acc, gyr = gyr, fs = frequency)
		return temp_estimation

	elif dim == '9D':
		print('RIANN is not a 9D filter')
		return 0

	else:
		pass 


def mah_filter(dim, gyr, acc, mag, frequency, q0 = None, params = None):
	""" Return 6D or 9D estimated orientation using MAH
	"""
	if dim == '6D':
		if params == None:
			temp_estimation = Mahony(gyr = gyr, acc = acc, frequency = frequency, q0 = q0[0])
		else:
			temp_estimation = Mahony(gyr = gyr, acc = acc, frequency = frequency, k_P = params[0], k_I = params[1], q0 = q0[0])

	elif dim == '9D':
		if params == None:
			temp_estimation = Mahony(gyr = gyr, acc = acc, mag = mag, frequency = frequency)
		else:
			temp_estimation = Mahony(gyr = gyr, acc = acc, mag = mag, frequency = frequency, k_P = params[0], k_I = params[1])

	else:
		pass

	return temp_estimation


def mad_filter(dim, gyr, acc, mag, frequency, q0 = None, params = None):
	""" Return 6D or 9D estimated orientation using MAD
	"""
	if dim == '6D':
		if params == None:
			temp_estimation = Madgwick(gyr = gyr, acc = acc, frequency = frequency, q0 = q0[0])
		else:
			temp_estimation = Madgwick(gyr = gyr, acc = acc, frequency = frequency, gain = params[0], q0 = q0[0])

	elif dim == '9D':
		if params == None:
			temp_estimation = Madgwick(gyr = gyr, acc = acc, mag = mag, frequency = frequency)
		else:
			temp_estimation = Madgwick(gyr = gyr, acc = acc, mag = mag, frequency = frequency, gain = params[0])

	else:
		pass

	return temp_estimation


def comp_filter(dim, gyr, acc, mag, frequency, q0 = None, params = None):
	""" Return 6D or 9D estimated orientation using COMP
	"""
	if dim == '6D':
		if params == None:
			temp_estimation = Complementary(gyr = gyr, acc = acc, frequency = frequency, q0 = q0[0])
		else:
			temp_estimation = Complementary(gyr = gyr, acc = acc, frequency = frequency, gain = params[0], q0 = q0[0])

	elif dim == '9D':
		if params == None:
			temp_estimation = Complementary(gyr = gyr, acc = acc, mag = mag, frequency = frequency)
		else:
			temp_estimation = Complementary(gyr = gyr, acc = acc, mag = mag, frequency = frequency, gain = params[0])

	else:
		pass

	return temp_estimation


def ekf_filter(dim, gyr, acc, mag, frequency, q0 = None, params = None):
	""" Return 6D or 9D estimated orientation using EKF
	"""
	# pitt_wmm     = wmm.WMM(latitude=40.44, longitude=-80, height=0.233)
	# PITT_MAG_REF = np.array([pitt_wmm.Y, pitt_wmm.X, -pitt_wmm.Z])

	if dim == '6D':
		if params == None:
			temp_estimation = EKF(gyr = gyr, acc = acc, frequency = frequency, frame='ENU', q0 = q0[0])
		else:
			temp_estimation = EKF(gyr = gyr, acc = acc, frequency = frequency, frame='ENU', noises = np.array([params[0]**2, params[1]**2, params[2]**2]), q0 = q0[0])

	elif dim == '9D':
		if params == None:
			temp_estimation = EKF(gyr = gyr, acc = acc, mag = mag, frequency = frequency, frame='ENU')
		else:
			temp_estimation = EKF(gyr = gyr, acc = acc, mag = mag, frequency = frequency, frame='ENU', noises = np.array([params[0]**2, params[1]**2, params[2]**2]))

	else:
		pass

	return temp_estimation


def saam_filter(dim, gyr, acc, mag, frequency, params = None):
	""" Return 6D estimated orientation using SAAM
	"""
	if dim == '6D':
		if params != None:
			min_cutoff = params[0]
			beta       = params[1]

			acc[:, 0] = smoothing_gyro_free(1*acc[:, 0], min_cutoff, beta, frequency)
			acc[:, 1] = smoothing_gyro_free(1*acc[:, 1], min_cutoff, beta, frequency)
			acc[:, 2] = smoothing_gyro_free(1*acc[:, 2], min_cutoff, beta, frequency)

		temp_estimation = saam.SAAM(acc = acc, mag = mag)

		return temp_estimation

	elif dim == '9D':
		print('SAAM is not a 9D filter')
		return 0

	else:
		pass 


def smoothing_gyro_free(sig, min_cutoff, beta, frequency):
	""" Smoothing with one euro filter
	"""
	t = 0
	x0 = 1*sig[0]
	# min_cutoff = 0.004
	# beta = 0.002
	one_euro_filter = OneEuroFilter(
		t, x0,
		min_cutoff=min_cutoff,
		beta=beta
	)

	filtered_sig = []
	for i in range(sig.shape[0]):
		t += 1.0/frequency
		temp_ = one_euro_filter(t, sig[i])
		filtered_sig.append(temp_) 
	
	filtered_sig = np.array(filtered_sig)
	
	return filtered_sig


def quest_filter(dim, gyr, acc, mag, frequency, params = None):
	""" Return 6D estimated orientation using QUEST
	"""
	if dim == '6D':
		if params == None:
			temp_estimation = quest.QUEST(acc = acc, mag = mag)
		else:
			min_cutoff = params[0]
			beta       = params[1]

			if params[2] == True:
				acc[:, 0] = smoothing_gyro_free(1*acc[:, 0], min_cutoff, beta, frequency)
				acc[:, 1] = smoothing_gyro_free(1*acc[:, 1], min_cutoff, beta, frequency)
				acc[:, 2] = smoothing_gyro_free(1*acc[:, 2], min_cutoff, beta, frequency)

			temp_estimation = quest.QUEST(acc = acc, mag = mag, weights = np.array(params[3:5]))

		return temp_estimation

	elif dim == '9D':
		print('QUEST is not a 9D filter')
		return 0

	else:
		pass 


def flae_filter(dim, gyr, acc, mag, frequency, params = None):
	""" Return 6D estimated orientation using FLAE
	"""
	if dim == '6D':
		if params == None:
			temp_estimation = flae.FLAE(acc = acc, mag = mag)
		else:
			min_cutoff = params[0]
			beta       = params[1]

			if params[2] == True:
				acc[:, 0] = smoothing_gyro_free(1*acc[:, 0], min_cutoff, beta, frequency)
				acc[:, 1] = smoothing_gyro_free(1*acc[:, 1], min_cutoff, beta, frequency)
				acc[:, 2] = smoothing_gyro_free(1*acc[:, 2], min_cutoff, beta, frequency)

			temp_estimation = flae.FLAE(acc = acc, mag = mag, method = params[3], weights = np.array(params[4:6]))
			
		return temp_estimation

	elif dim == '9D':
		print('FLAE is not a 9D filter')
		return 0

	else:
		pass


def fqa_filter(dim, gyr, acc, mag, frequency, params = None):
	""" Return 6D estimated orientation using FQA
	"""
	# Wn   = 0.5*2/40
	# b, a = signal.butter(4, Wn, btype = 'low')
	# ax   = signal.filtfilt(b, a, 1*acc[:, 0], axis = 0)
	# ay   = signal.filtfilt(b, a, 1*acc[:, 1], axis = 0)
	# az   = signal.filtfilt(b, a, 1*acc[:, 2], axis = 0)
	# filtered_acc = np.array([ax, ay, az]).T

	if dim == '6D':
		if params == 'None':
			temp_estimation = fqa.FQA(acc = acc, mag = mag)
		else:
			temp_estimation = fqa.FQA(acc = acc, mag = mag, mag_ref = params)
		return temp_estimation

	elif dim == '9D':
		print('FQA is not a 9D filter')
		return 0

	else:
		pass


