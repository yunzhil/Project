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

def segment(file, col_no, segment_area=(1000, 9000), min_height_low=-1, min_height_high=1, max_height_low=-1, max_height_high=1):
    raw_mocap = pd.read_csv(file, sep='\s')
    mocap_data = raw_mocap[5:].reset_index(drop=True)
    mocap_data.columns = pd.RangeIndex(mocap_data.columns.size)
    mocap_dt = mocap_data[0]
    mocap_full = []

    for row in mocap_dt:
        mocap_full.append(row.split(','))
    mocap_df = pd.DataFrame(mocap_full)
    mocap_df.drop(columns=[0], inplace=True)
    marker1 = pd.DataFrame({0:mocap_df[2], 1:mocap_df[3], 2:mocap_df[4]})
    marker1 = marker1.astype(float)
    marker1.plot()
    marker1 = preprocessing_mocap.lowpass_filter_mocap(marker1, constant_mocap.MOCAP_SAMPLING_RATE,
                                                            constant_mocap.FILTER_CUTOFF_MOCAP,
                                                            constant_mocap.FILTER_ORDER)
    marker1 = marker1[segment_area[0]:segment_area[1]] #2345:3546
    marker1.reset_index(inplace=True)
    x = marker1['index']
    #fp = findpeaks()
    #results = fp.fit(marker1[2])
    #fp.plot()
    # height = [min_height_low, min_height_high], 
    # height = [max_height_low, max_height_high], 
    minima, _ = find_peaks(-1*marker1[col_no], distance = constant_mocap.MOCAP_SAMPLING_RATE * 0.5)
    #min = marker1[col_no][minima].nsmallest(5).index
    #maxima = find_peaks_cwt(np.array(list(marker1[col_no])), 50)
    maxima, _ = find_peaks(marker1[col_no], distance = constant_mocap.MOCAP_SAMPLING_RATE * 0.5)
    #x = np.linspace(0, len(marker1[2]), len(marker1[2]))
    save_file = r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\segments' + '\\' + str(col_no) + file[-18:]
    plt.plot(x, marker1[col_no])
    plt.plot(x[minima], marker1[col_no][minima], 'x', label='mins')
    ##plt.plot(x[maxima], marker1[col_no][maxima], 'o', label='max')
    plt.legend()
    plt.savefig(save_file[:-4])
    plt.show(block = False)
    plt.pause(0.3)
    maxima, _ = find_peaks(marker1[col_no])
    peak_dict = {save_file: [minima, maxima]}
    peak_df = pd.DataFrame.from_dict(peak_dict)
    peak_df.to_csv(save_file)
    plt.close('all')
    ##plt.plot(x, marker1[col_no])
    ##plt.plot(x[minima], marker1[col_no][minima], 'x', label='mins')
    ##plt.plot(x[maxima], marker1[col_no][maxima], 'o', label='max')
    ##plt.show()
    #minpeaks = pd.Series(minima)
    #maxpeaks = pd.Series(maxima)
    #minima = minpeaks.nsmallest(5)
    #maxima = maxpeaks.nlargest(5)

#segment(r'C:\Users\kiddb\Downloads\eval_leg_swing_001.csv', 0, 2, 3, -1, 0)
# 0 - 100 and interpolate. Then average each point (from 0 to 100 of the 5 reps) to get gait cycle (don't forget std dev) or repetition cycle.
# try to plot error cruve
#extract the minima and maxima and define a cycle as minima to minima (edge cases where minima is outside of cycle?)