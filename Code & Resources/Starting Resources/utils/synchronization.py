# name: synchronization.py
# description: Synchronize mocap and IMU data
# author: Vu Phan
# date: 2023/06/05


import numpy as np
import pandas as pd
import quaternion
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

import sys, os
sys.path.append('/path/to/IMU_Kinematics_Comparison_v2')

from utils.mt import constant_mt
from utils.mt.preprocessing_mt import load_data_mt
from utils.common import get_rmse


# --- Get vertical acceleration from a specific IMU --- #
def get_vertical_acc_mt(one_imu_data):
    ''' Get vertical acceleration of an IMU data
    Args:
        + one_imu_data (pd.DataFrame): data from the desired sensor

    Returns:
        + vertical_acc_mt (np.array): vertical acceleration (gravity relatively removed)
    '''
    vertical_acc_mt = 1*one_imu_data['Acc_X'].to_numpy()
    vertical_acc_mt -= constant_mt.EARTH_G_ACC

    return vertical_acc_mt

# --- Get vertical acceleration from a spcific marker --- #
def get_vertical_acc_mocap(one_marker_data):
    ''' Get vertical acceleration of a marker

    Args:
        + one_marker_data (pd.DataFrame): vertical motion of a marker in the mocap data

    Returns:
        + vertical_acc_mocap (np.array): vertical acceleration
    '''
    vertical_acc_mocap = 1*one_marker_data.to_numpy()
    vertical_acc_mocap = np.diff(vertical_acc_mocap)/(1.0/constant_mt.MT_SAMPLING_RATE)
    vertical_acc_mocap = np.diff(vertical_acc_mocap)/(1.0/constant_mt.MT_SAMPLING_RATE)

    return vertical_acc_mocap

# --- Identify the hop period with mocap data --- #
def get_hop_id_mocap(one_mocap_data):
    ''' Get hop id from mocap

    Args:
        + one_marker_data (pd.DataFrame): vertical motion of a marker in the mocap data

    Returns:
        + hop_id_mocap (int): id of the mid hop
    '''    
    possible_id, _ = find_peaks(one_mocap_data, height = 5)

    hop_id_mocap = 0
    peak_count   = 0
    while hop_id_mocap < 200:
        hop_id_mocap = 1*possible_id[peak_count]
        peak_count  += 1

    return hop_id_mocap

# --- Get information for sync'ing --- #
def get_sync_info(pelvis_vertical_acc_mt, pelvis_vertical_acc_mocap, hop_id_mocap, window = 120, iters = 1500):
    ''' Get information for sync'ing IMU and mocap data

    Args:
        + pelvis_vertical_acc_mt (np.array): vertical acceleration of the pelvis sensor (gravity relatively removed)
        + pelvis_vertical_acc_mocap (np.array): vertical acceleration of the pelvis sensor
        + window, iters (int): parameters for matching IMU and mocap data

    Returns:
        + first_start (str): 'imu' or 'mocap'
        + shifting_id (int): shifting amount for IMU or mocap to sync
    '''
    shifting_id = 0
    prev_err    = 999

    error = []
    for i in range(iters):
        start_mocap = hop_id_mocap - int(window/2)
        stop_mocap = hop_id_mocap + int(window/2)
        if i > start_mocap:
            break
        start_imu = start_mocap - i
        stop_imu = stop_mocap - i
        curr_err = get_rmse(pelvis_vertical_acc_mocap[start_mocap:stop_mocap], pelvis_vertical_acc_mt[start_imu:stop_imu])
        error.append(curr_err)

        if curr_err < prev_err:
            shifting_id = i + 2 # due to double derivative for mocap acceleration
            prev_err = curr_err

    mocap_error = min(error)
    mocap_shifting_id = shifting_id

    # import matplotlib.pyplot as plt
    # breakpoint()

    error = []
    for i in range(iters):
        start_mocap = hop_id_mocap - int(window/2)
        stop_mocap = hop_id_mocap + int(window/2)
        start_imu = start_mocap + i
        stop_imu = stop_mocap + i
        curr_err = get_rmse(pelvis_vertical_acc_mocap[start_mocap:stop_mocap], pelvis_vertical_acc_mt[start_imu:stop_imu])
        error.append(curr_err)

        if curr_err < prev_err:
            if i >=2:
                shifting_id = i - 2 # due to double derivative for mocap acceleration
            else:
                shifting_id = 0
            prev_err = curr_err

    mt_err = min(error)
    mt_shifting_id = shifting_id

    # breakpoint()

    if mocap_error < mt_err:
        shifting_id = mocap_shifting_id
        first_start = 'mocap'
    else:
        shifting_id = mt_shifting_id
        first_start = 'imu'

    print(first_start)
    print(shifting_id)

    # breakpoint()

    return first_start, shifting_id


def sync_ja_mocap_mt(mocap_ja, mocap_data, imu_ja, imu_data, iters = 1500):
    """ Sync mocap and IMU data

    Params:
        mocap_ja: mocap-based joint angles | dict of np.array
        mocap_data (pd.DataFrame): mocap data of the main task
        imu_ja: IMU-based joint angles | dict of np.array
        imu_data (dict of pd.DataFrame): IMU data of the main task

    Returns:
        sync_mocap_ja: mocap-based joint angles sync'ed with IMU-based | dict of np.array
        sync_imu_ja: IMU-based joint angles sync'ed with mocap-based | dict of np.array
    """
    pelvis_vertical_acc_mt    = get_vertical_acc_mt(imu_data['pelvis'])
    pelvis_vertical_acc_mocap = get_vertical_acc_mocap(mocap_data['RPS2 Y'])
    hop_id_mocap              = get_hop_id_mocap(pelvis_vertical_acc_mocap[0:int(len(pelvis_vertical_acc_mocap)/2)])

    first_start, shifting_id = get_sync_info(pelvis_vertical_acc_mt, pelvis_vertical_acc_mocap, hop_id_mocap, iters = iters)

    sync_mocap_ja = {}
    sync_imu_ja   = {}

    if first_start == 'mocap':
        for jk in imu_ja.keys():
            sync_mocap_ja[jk] = 1*mocap_ja[jk][shifting_id:-1]
            sync_imu_ja[jk]   = 1*imu_ja[jk]
            
    else:
        for jk in imu_ja.keys():
            sync_mocap_ja[jk] = 1*mocap_ja[jk]
            sync_imu_ja[jk]   = 1*imu_ja[jk][shifting_id:-1]

    return sync_mocap_ja, sync_imu_ja
