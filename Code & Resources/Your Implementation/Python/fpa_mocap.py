# name: fpa_mocap.py
# description: obtain FPA using mocap data
# author: Vu Phan
# date: 2024/04/17


import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import signal
from scipy.signal import find_peaks

import visualizer


# --- Remove noisy peaks --- #
def remove_noisy_peaks_mocap(raw_index, raw_value, task):
    ''' Keeps peaks during walking only

    Args:
        + raw_index, raw_value (np.array): outputs from np.find_peaks

    Returns:
        + index, value (np.array): index and value of a specific gait event during walking
    '''
    alpha = 0.5

    value_sorted = 1*raw_value
    value_sorted.sort()
    upper_bound  = np.mean(value_sorted[::3])
    lower_bound  = np.mean(value_sorted[-3::])

    threshold   = alpha*upper_bound + (1 - alpha)*lower_bound
    selected_id = np.where(raw_value < threshold)[0]
    index       = 1*raw_index[selected_id]
    value       = 1*raw_value[selected_id]

    return index, value


# --- Identify heel-contact and toe-off events from the mocap data --- #
# Method using the height of heel and toe markers
def ge_heel_toe_height(marker_traj, fs = 100, vis = False):
    ''' Obtain gait events from the height of heel and toe markers

    Args:
        + marker_traj (dict of np.array): dictionary of marker trajectories
        + fs (int): sampling rate of the mocap data
        + vis (bool): visualization flag
    '''
    min_peak_distance_hc = 1*fs
    min_peak_distance_to = 1*fs
    gait_events = {}

    heel_marker_y                = marker_traj['heel_marker_y']
    temp_hc_index, temp_hc_value = find_peaks(-1*heel_marker_y, height = 0.03, distance = min_peak_distance_hc)
    hc_index                     = 1*temp_hc_index[2:-2]
    hc_value                     = -1*temp_hc_value['peak_heights'][2:-2]
    # hc_index, hc_value           = remove_noisy_peaks_mocap(hc_index, hc_value, 'walking')

    toe_marker_y                 = marker_traj['toe_marker_y']
    temp_to_index, temp_to_value = find_peaks(-1*toe_marker_y, height = 0.01, distance = min_peak_distance_to)
    to_index                     = 1*temp_to_index
    to_value                     = -1*temp_to_value['peak_heights']
    # to_index, to_value           = remove_noisy_peaks_mocap(to_index, to_value, 'walking')

    gait_events['hc_index'] = hc_index
    gait_events['hc_value'] = hc_value
    gait_events['to_index'] = to_index
    gait_events['to_value'] = to_value

    if vis:
        visualizer.plot_gait_events_mocap(heel_marker_y, toe_marker_y, gait_events)

    return gait_events


def get_fpa(data_mocap, side):
    ''' Constrain the gait events (remove unwanted peaks)
    '''
    marker_mt   = data_mocap[[side[0].upper() + 'MT2 X', side[0].upper() + 'MT2 Z']].to_numpy()
    marker_heel = data_mocap[[side[0].upper() + 'CAL X', side[0].upper() + 'CAL Z']].to_numpy()
    foot_vec    = marker_mt - marker_heel

    fpa = np.arctan([foot_vec[:, 0]/foot_vec[:, 1]])[0]
    fpa = np.rad2deg(fpa)

    if side == 'right':
        fpa = -1*fpa

    return fpa


def get_fpa_stance(fpa_all, event_mocap, vis = False):
    ''' Get FPA from 15% - 50% of stance phase for each step
    '''
    fpa = []
    fpa_period = []
    for i in range(len(event_mocap['hc_index']) - 1):
        hc      = event_mocap['hc_index'][i]
        hc_next = event_mocap['hc_index'][i + 1]
        
        temp = np.where(event_mocap['to_index'] > hc)[0]
        if len(temp) != 0:
            to_id = temp[0]
            to    = event_mocap['to_index'][to_id]
            
            if to < hc_next:
                # fpa_stance = fpa_all[hc:to]
                # fpa_stance = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(fpa_stance)), fpa_stance)
                # fpa.append(np.mean(fpa_stance[15:50]))
                period = np.arange(hc + int(0.15*(to - hc)), hc + int(0.5*(to - hc)), 1)
                fpa.append(np.mean(fpa_all[period]))
                fpa_period.append(period)

    fpa = np.array(fpa)

    if vis:
        visualizer.plot_fpa_stance(fpa_all, fpa_period, fpa)

    return fpa


# --- Remove outliers in FPA (due to tuning in overground walking) --- #
def remove_fpa_outliers(fpa):
    ''' Remove outliers in FPA due to tuning in overground walking
    '''
    upper_bound = np.mean(fpa) + 1*np.std(fpa)
    lower_bound = np.mean(fpa) - 1*np.std(fpa)

    selected_id = np.where((fpa < upper_bound) & (fpa > lower_bound))[0]
    fpa_cleaned = fpa[selected_id]

    return fpa_cleaned


