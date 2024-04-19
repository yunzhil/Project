# name: segmentation.py
# description: Segment joint angles to reps (for exercises) and gait cycles (for walking/running)
# author: Kevin Bui
# date: 2024/03/18
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from util.mocap import preprocessing_mocap, constant_mocap

def get_data_mocap(file_path):
    ''' Obtain the mocap data 

    Args:
        + subject (int): subject id
        + task (str): task being performed, e.g., static, walking, squat, etc.

    Returns:
        + data_mocap (pd.DataFrame): mocap data
    '''

    mocap_data = pd.read_csv(file_path, skiprows = 3, low_memory = False)
    mocap_data = mocap_data.iloc[:, 1:]
    mocap_data = mocap_data.iloc[2:, :]

    names_pos = list(mocap_data.columns)
    names_pos = [name.split(':')[1][0:4] for name in names_pos[1:]]
    names_pos = [''] + names_pos
    names_ax  = mocap_data.iloc[0, :]
    names     = []

    for i in range(len(names_pos)):
        names.append(names_pos[i] + ' ' + names_ax[i])
    names[0] = 'Time'

    mocap_data = mocap_data.iloc[1:, :]
    mocap_data.columns = names

    mocap_data = mocap_data.reset_index()
    mocap_data = mocap_data.iloc[:, 1:]
    mocap_data = mocap_data.astype('float64')

    return mocap_data

def segment(file, exercise = '', segment=[], segment_area=(1000, 9000), min_height_low=-1, min_height_high=1, max_height_low=-1, max_height_high=1):
    subject = 'subject_001'
    raw_mocap = pd.read_csv(file, sep='\s')
    cols = pd.Series(raw_mocap['Format'][1].split(',')).iloc[2:].str[-4:]
    cols.reset_index(drop=True, inplace=True)
    axis = pd.Series(raw_mocap['Version,1.23,Take'][4].split(',')[1:])
    header = cols+axis
    mocap_data_list = []

    for row in raw_mocap['Format'][5:]:
        mocap_data_list.append(row.split(','))
    mocap_df = pd.DataFrame(mocap_data_list)
    mocap_df.columns = ['Frame', 'Time', *header]
    mocap_df = get_data_mocap(file)
    if segment:
        segment.insert(0, 'Time')
        marker = mocap_df[segment]
    else:
        marker = mocap_df[header]
    try:
        marker = marker.astype(float)
    except ValueError:
        marker = marker.replace(r'^\s*$', np.nan, regex=True)
        marker = marker.astype(float)
    #marker.plot()
    #plt.show()
    marker = preprocessing_mocap.resample_mocap(marker, 60)
    marker = preprocessing_mocap.lowpass_filter_mocap(marker, constant_mocap.MOCAP_SAMPLING_RATE,
                                                            constant_mocap.FILTER_CUTOFF_MOCAP,
                                                            constant_mocap.FILTER_ORDER)
    
    marker[segment[1:3]].plot()
    roi = marker[segment_area[0]:segment_area[1]]
    marker.reset_index(drop=True, inplace=True)
    ##x = marker['index']
    minima, _ = find_peaks(-1*roi[segment[1]], prominence = 0.5, distance = constant_mocap.MOCAP_SAMPLING_RATE * 0.5)
    maxima, _ = find_peaks(marker.iloc[2], distance = constant_mocap.MOCAP_SAMPLING_RATE * 0.5)
    #save_file = r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\segments' + '\\' + str(1) + file[-18:]
    save_file = ''
    for ind, seg in enumerate(segment):
        if ind != 0:
            save_file += seg
    plt.plot(marker[segment[1:]])
    plt.plot(roi[segment[1]].iloc[minima], 'x', label='mins')
    ##plt.plot(x[maxima], marker1[col_no][maxima], 'o', label='max')
    plt.legend()
    plt.title(exercise)
    plt.savefig(r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\Mocap' + '\\' + subject + '\\segment\\' + save_file + '_' + exercise[:-4])
    #plt.show(block = False)
    #plt.pause(30)
    maxima, _ = find_peaks(marker.iloc[1])
    #peak_dict = {save_file: [minima, maxima]}
    #peak_df = pd.DataFrame.from_dict(peak_dict)
    #peak_df.to_csv(save_file)
    plt.close('all')


#segment(r'C:\Users\kiddb\Downloads\eval_leg_swing_001.csv', segment=['LCALX', 'LCALY', 'LCALZ'], segment_area=(1550, 2500), min_height_low=2, min_height_high=3, max_height_low=3, max_height_high=-1)
# 0 - 100 and interpolate. Then average each point (from 0 to 100 of the 5 reps) to get gait cycle (don't forget std dev) or repetition cycle.
# try to plot error cruve
#extract the minima and maxima and define a cycle as minima to minima (edge cases where minima is outside of cycle?)
