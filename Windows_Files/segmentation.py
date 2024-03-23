# name: segmentation.py
# description: Segment joint angles to reps (for exercises) and gait cycles (for walking/running)
# author: Kevin Bui
# date: 2024/03/18
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import numpy as np

#def segment(min, max):
#    pass

raw_mocap = pd.read_csv(r'C:\Users\kiddb\Downloads\eval_hip_adduction_001.csv', sep='\s')
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
marker1 = marker1[2345:3546]
marker1.reset_index(inplace=True)
x = marker1['index']
#fp = findpeaks()
#results = fp.fit(marker1[2])
#fp.plot()
minima, _ = find_peaks(-1*marker1[2])
min = marker1[2][minima].nsmallest(5).index
maxima = find_peaks_cwt(np.array(list(marker1[2])), 50)
#x = np.linspace(0, len(marker1[2]), len(marker1[2]))
plt.plot(x, marker1[2])
plt.plot(x[min], marker1[2][min], 'x', label='mins')
plt.plot(x[maxima], marker1[2][maxima], 'o', label='max')
plt.legend()
plt.show()
# 0 - 100 and interpolate. Then average each point (from 0 to 100 of the 5 reps) to get gait cycle (don't forget std dev) or repetition cycle.
# try to plot error cruve
#extract the minima and maxima and define a cycle as minima to minima (edge cases where minima is outside of cycle?)