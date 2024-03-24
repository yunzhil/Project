# name: gait_params_mocap.py
# description: Compute gait parameters using mocap (as the ground truth)
# author: Vu Phan
# date: 2024/01/05


import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import signal
from scipy.signal import find_peaks

import constants_meta
from utils import visualizer
from utils.mt import constants_mt


# --- Remove noisy peaks --- #
def remove_noisy_peaks_mocap(raw_index, raw_value, task):
    ''' Keeps peaks during walking only

    Args:
        + raw_index, raw_value (np.array): outputs from np.find_peaks

    Returns:
        + index, value (np.array): index and value of a specific gait event during walking
    '''
    if task == 'treadmill_walking':
        threshold_1 = np.mean(raw_value) + 0.1*np.std(raw_value)
        threshold_2 = np.mean(raw_value) - 2*np.std(raw_value)
    elif task == 'walking':
        threshold_1 = np.mean(raw_value) + 0.5*np.std(raw_value)
        threshold_2 = np.mean(raw_value) - 2*np.std(raw_value)

    id1         = np.where(raw_value < threshold_1)[0]
    id2         = np.where(raw_value > threshold_2)[0]
    selected_id = np.intersect1d(id1, id2)
    index       = 1*raw_index[selected_id]
    value       = 1*raw_value[selected_id]

    return index, value


def get_hc_n_to_mocap(heel_marker_y, toe_marker_y, task, fs = 100, remove = 10, vis = False):
    ''' Obtain heel contact and toe-off events

    Args:
        + heel_marker_y, toe_marker_y (np.array): vertical position of heel & toe markers
        + task (str): 'walking' or 'treadmill_walking
        + fs (int): sampling rate
        + remove (int): number of sampled to be removed at the beginning and end

    Returns:
        + gait_events (dict of np.array): index and value arrays of heel strike and toe-offs
    '''
    min_peak_distance_hc = fs*0.5
    min_peak_distance_to = fs*0.5
    gait_events = {'hc_index': [], 'hc_value': [], 'to_index': [], 'to_value': []}

    temp_hc_index, temp_hc_value = find_peaks(-1*heel_marker_y, height = [-1, 0], distance = min_peak_distance_hc)
    hc_index, hc_value           = remove_noisy_peaks_mocap(temp_hc_index, -1*temp_hc_value['peak_heights'], task)

    temp_to_index, temp_to_value = find_peaks(-1*toe_marker_y, height = [-1, 0], distance = min_peak_distance_to)
    # temp_to_index, temp_to_value = remove_noisy_peaks_mocap(temp_to_index, -1*temp_to_value['peak_heights'], task)
    # temp_to_value                = -1*temp_to_value['peak_heights']
    # near_to_index, to_value      = correct_near_to_mocap(hc_index, temp_to_index, temp_to_value)
    # to_index                     = correct_to_mocap(near_to_index, toe_marker_y, fs)
    to_index                     = 1*temp_to_index
    to_value                     = -1*temp_to_value['peak_heights']

    gait_events['hc_index'] = 1*hc_index
    gait_events['hc_value'] = 1*hc_value
    gait_events['to_index'] = 1*to_index
    gait_events['to_value'] = 1*to_value

    if vis == True:
        visualizer.plot_gait_events_mocap(heel_marker_y, toe_marker_y, gait_events)

    return gait_events
