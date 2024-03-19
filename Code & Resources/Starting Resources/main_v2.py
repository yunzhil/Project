# name: main_v2.py
# description: Updated IMU kinematics estimation
# author: Vu Phan
# date: 2024/01/23


# TODO: Sensor placement with S2S calibration - using Xsens filter
# TODO: Comparison between sensor fusion algorithms (with S2S calibration) - using different filters
# TODO: Comparison between unconstrained (only S2S calibration) vs. constrained kinematics estimation - using Xsens and the other best filter

# NOTE: Need to find a way to align IMU results with mocap results (i.e., remove the offset)


import pandas as pd 
import numpy as np 
import quaternion

from utils import va, synchronization, common, constant_common
from utils.mt import constant_mt, preprocessing_mt, calibration_mt, ik_mt
from utils.mocap import constant_mocap, preprocessing_mocap, ik_mocap


# Set up the configurations here
f_type = 'Xsens'
dim    = '9D' # only 9D at the moment

if f_type == 'MAD':
    f_params = [0.1]
elif f_type == 'MAH':
    f_params = [0.4, 0.3]
elif f_type == 'EKF':
    f_params = [0.9, 0.9, 0.9]
elif f_type == 'VQF':
    f_params = [2, 10]
else:
    f_params = None


selected_setup = 'mid' # sensor placement, i.e., 'mid' (for main analysis), 'high', 'low', or 'front'

# MAPPING_TASK_TO_ID = {'static':            0,
#                      'walking':           1,
#                      'treadmill_walking': 2,
#                      'treadmill_running': 3,
#                      'lat_step':          4,
#                      'step_up_down':      5,
#                      'drop_jump':         6,
#                      'cmj':               7,
#                      'squat':             8,
#                      'step_n_hold':       9,
#                      'sls':               10,
#                      'sts':               11}

selected_task  = 'static'
subject        = 3
print('*** Subject ' + str(subject))
print('*** Sensor setup ' + selected_setup)

# --- IMU data --- #
# Sensor configuration for analysis
sensor_config  = {'pelvis': 'PELVIS', 
                  'foot_r': 'FOOT_R', 'shank_r': 'SHANK_R_' + selected_setup.upper(), 'thigh_r': 'THIGH_R_' + selected_setup.upper(),
                  'foot_l': 'FOOT_L', 'shank_l': 'SHANK_L_' + selected_setup.upper(), 'thigh_l': 'THIGH_L_' + selected_setup.upper()}


# Data for calibration (DO NOT CHANGE HERE)
task_static     = 'static'
data_static_mt  = preprocessing_mt.get_all_data_mt(subject, task_static, sensor_config)
data_static_mt_ = preprocessing_mt.match_data_mt(data_static_mt) # data after matching
task_walking    = 'treadmill_walking' # to calibrate thighs, shanks, and feet
data_walking_mt = preprocessing_mt.get_all_data_mt(subject, task_walking, sensor_config)
task_squat      = 'cmj' # to calibrate pelvis
data_squat_mt   = preprocessing_mt.get_all_data_mt(subject, task_squat, sensor_config)

# Data for analysis
data_main  = preprocessing_mt.get_all_data_mt(subject, selected_task, sensor_config)
data_main_ = preprocessing_mt.match_data_mt(data_main) # data after matching

# Sensor-to-body alignment
if selected_setup == 'front':
    walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Y'].to_numpy())
else:
    walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Z'].to_numpy())
    # plot walking_period
    # plt.plot(data_walking_mt['shank_r']['Gyr_Z'])
squat_period = [0, data_squat_mt['pelvis']['Gyr_Y'].shape[0]]
seg2sens     = calibration_mt.sensor_to_segment_mt(data_static_mt, data_walking_mt, walking_period, data_squat_mt, squat_period, selected_setup)

# Obtain sensor orientations
# Get static offset for correction
static_orientation_mt = ik_mt.get_imu_orientation_mt(data_static_mt_, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim, params = f_params)
static_ja_mt          = ik_mt.get_all_ja_mt(seg2sens, static_orientation_mt)
# breakpoint()
# TODO: get joint angles during task
main_orientation_mt = ik_mt.get_imu_orientation_mt(data_main_, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim, params = f_params)
main_ja_mt          = ik_mt.get_all_ja_mt(seg2sens, main_orientation_mt)
# breakpoint()
for jk in main_ja_mt.keys():
    offset         = np.mean(static_ja_mt[jk])
    main_ja_mt[jk] = main_ja_mt[jk] - offset
    

# --- Mocap data --- #
# Data for calibration
data_static_mocap     = preprocessing_mocap.get_data_mocap(subject, task_static)
data_static_mocap     = data_static_mocap.interpolate(method = 'cubic')
data_static_mocap     = data_static_mocap.fillna(value = 999)
data_static_mocap_avg = preprocessing_mocap.get_avg_data(data_static_mocap)

# Data for analysis
data_main_mocap = preprocessing_mocap.get_data_mocap(subject, selected_task)
data_main_mocap = data_main_mocap.interpolate(method = 'cubic')
data_main_mocap = data_main_mocap.fillna(value = 999)
data_main_mocap = preprocessing_mocap.lowpass_filter_mocap(data_main_mocap, constant_mocap.MOCAP_SAMPLING_RATE,
                                                           constant_mocap.FILTER_CUTOFF_MOCAP,
                                                           constant_mocap.FILTER_ORDER) # filter
data_main_mocap = preprocessing_mocap.resample_mocap(data_main_mocap, constant_mt.MT_SAMPLING_RATE) # downsample

# Get orientation
static_orientation_mocap_avg = ik_mocap.get_orientation_mocap(data_static_mocap_avg, cluster_use = True)
static_orientation_mocap     = ik_mocap.get_orientation_mocap(data_static_mocap, cluster_use = True)
main_orientation_mocap       = ik_mocap.get_orientation_mocap(data_main_mocap, cluster_use = True)
cal_orientation_mocap        = ik_mocap.calibration_mocap(static_orientation_mocap_avg, cluster_use = True)

# Get joint angles
static_ja_mocap = ik_mocap.get_all_ja_mocap(cal_orientation_mocap, static_orientation_mocap, cluster_use = True)
main_ja_mocap   = ik_mocap.get_all_ja_mocap(cal_orientation_mocap, main_orientation_mocap, cluster_use = True)

for jk in main_ja_mocap.keys():
    offset         = np.mean(static_ja_mocap[jk])
    main_ja_mocap[jk] = main_ja_mocap[jk] - offset


# --- Sync mocap and IMU --- #
if selected_task in ['lat_step', 'step_up_down', 'drop_jump', 'cmj', 'squat', 'step_n_hold', 'sls', 'sts']:
    iters = 600 # 800 previously
else:
    iters = 1500
# main_ja_mocap, main_ja_mt = synchronization.sync_ja_mocap_mt(main_ja_mocap, data_main_mocap, main_ja_mt, data_main_, iters = iters)



import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows = 5, ncols = 2, sharex = True)
ax[0, 0].plot(main_ja_mocap['hip_adduction_l'])
ax[1, 0].plot(main_ja_mocap['hip_rotation_l'])
ax[2, 0].plot(main_ja_mocap['hip_flexion_l'])
ax[3, 0].plot(main_ja_mocap['knee_flexion_l'])
ax[4, 0].plot(main_ja_mocap['ankle_angle_l'])

ax[0, 1].plot(main_ja_mocap['hip_adduction_r'])
ax[1, 1].plot(main_ja_mocap['hip_rotation_r'])
ax[2, 1].plot(main_ja_mocap['hip_flexion_r'])
ax[3, 1].plot(main_ja_mocap['knee_flexion_r'])
ax[4, 1].plot(main_ja_mocap['ankle_angle_r'])

ax[0, 0].plot(main_ja_mt['hip_adduction_l'])
ax[1, 0].plot(main_ja_mt['hip_rotation_l'])
ax[2, 0].plot(main_ja_mt['hip_flexion_l'])
ax[3, 0].plot(main_ja_mt['knee_flexion_l'])
ax[4, 0].plot(main_ja_mt['ankle_angle_l'])

ax[0, 1].plot(main_ja_mt['hip_adduction_r'])
ax[1, 1].plot(main_ja_mt['hip_rotation_r'])
ax[2, 1].plot(main_ja_mt['hip_flexion_r'])
ax[3, 1].plot(main_ja_mt['knee_flexion_r'])
ax[4, 1].plot(main_ja_mt['ankle_angle_r'])

plt.show()



# --- Compare IMU vs. mocap --- #
meta_fn = constant_common.IN_LAB_PATH + 'eval_period.xlsx'
meta_dt = pd.read_excel(meta_fn, sheet_name = 's' + str(subject))

start_ja_id = int(meta_dt['start'][constant_common.MAPPING_TASK_TO_ID[selected_task]])
stop_ja_id  = int(meta_dt['stop'][constant_common.MAPPING_TASK_TO_ID[selected_task]])



# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows = 5, ncols = 2, sharex = True)
# ax[0, 0].plot(main_ja_mocap['hip_adduction_l'][start_ja_id:stop_ja_id])
# ax[1, 0].plot(main_ja_mocap['hip_rotation_l'][start_ja_id:stop_ja_id])
# ax[2, 0].plot(main_ja_mocap['hip_flexion_l'][start_ja_id:stop_ja_id])
# ax[3, 0].plot(main_ja_mocap['knee_flexion_l'][start_ja_id:stop_ja_id])
# ax[4, 0].plot(main_ja_mocap['ankle_angle_l'][start_ja_id:stop_ja_id])

# ax[0, 1].plot(main_ja_mocap['hip_adduction_r'][start_ja_id:stop_ja_id])
# ax[1, 1].plot(main_ja_mocap['hip_rotation_r'][start_ja_id:stop_ja_id])
# ax[2, 1].plot(main_ja_mocap['hip_flexion_r'][start_ja_id:stop_ja_id])
# ax[3, 1].plot(main_ja_mocap['knee_flexion_r'][start_ja_id:stop_ja_id])
# ax[4, 1].plot(main_ja_mocap['ankle_angle_r'][start_ja_id:stop_ja_id])

# ax[0, 0].plot(main_ja_mt['hip_adduction_l'][start_ja_id:stop_ja_id])
# ax[1, 0].plot(main_ja_mt['hip_rotation_l'][start_ja_id:stop_ja_id])
# ax[2, 0].plot(main_ja_mt['hip_flexion_l'][start_ja_id:stop_ja_id])
# ax[3, 0].plot(main_ja_mt['knee_flexion_l'][start_ja_id:stop_ja_id])
# ax[4, 0].plot(main_ja_mt['ankle_angle_l'][start_ja_id:stop_ja_id])

# ax[0, 1].plot(main_ja_mt['hip_adduction_r'][start_ja_id:stop_ja_id])
# ax[1, 1].plot(main_ja_mt['hip_rotation_r'][start_ja_id:stop_ja_id])
# ax[2, 1].plot(main_ja_mt['hip_flexion_r'][start_ja_id:stop_ja_id])
# ax[3, 1].plot(main_ja_mt['knee_flexion_r'][start_ja_id:stop_ja_id])
# ax[4, 1].plot(main_ja_mt['ankle_angle_r'][start_ja_id:stop_ja_id])

# plt.show()



temp_ja_storage = []
for jk in main_ja_mt.keys():
    print(jk)
    temp_rmse = common.get_rmse(main_ja_mocap[jk][start_ja_id:stop_ja_id], main_ja_mt[jk][start_ja_id:stop_ja_id])
    temp_ja_storage.append(temp_rmse)

temp_ja_storage = np.array(temp_ja_storage)
temp_ja_storage = temp_ja_storage.reshape((1, len(temp_ja_storage)))
temp_ja_storage = pd.DataFrame(temp_ja_storage)
temp_ja_storage.to_csv('s' + str(subject) + '_' + selected_task + '_results.csv')







# Animate sensor orientation during standing (to check calibration)
# orientation_static = ik_mt.get_imu_orientation_mt(data_static_mt, 'RIANN', '9D')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib import animation

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(projection='3d')

origin = np.array([[0], [0], [0]])
x_axis = np.array([[1], [0], [0]])
y_axis = np.array([[0], [1], [0]])
z_axis = np.array([[0], [0], [1]])

def update(i):
    ax.clear()

    # Before calibration
    x_pelvis, y_pelvis, z_pelvis    = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['pelvis'][i]), x_axis, y_axis, z_axis)
    x_thigh_r, y_thigh_r, z_thigh_r = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['thigh_r'][i]), x_axis, y_axis, z_axis)
    x_shank_r, y_shank_r, z_shank_r = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['shank_r'][i]), x_axis, y_axis, z_axis)
    x_foot_r, y_foot_r, z_foot_r    = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['foot_r'][i]), x_axis, y_axis, z_axis)
    x_thigh_l, y_thigh_l, z_thigh_l = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['thigh_l'][i]), x_axis, y_axis, z_axis)
    x_shank_l, y_shank_l, z_shank_l = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['shank_l'][i]), x_axis, y_axis, z_axis)
    x_foot_l, y_foot_l, z_foot_l    = va.rotate_frame(quaternion.as_rotation_matrix(static_orientation_mt['foot_l'][i]), x_axis, y_axis, z_axis)

    va.add_frame_3D(ax, origin, x_pelvis, y_pelvis, z_pelvis, [0, 0, 6], True)
    va.add_frame_3D(ax, origin, x_thigh_r, y_thigh_r, z_thigh_r, [1, 0, 4], True)
    va.add_frame_3D(ax, origin, x_shank_r, y_shank_r, z_shank_r, [1, 0, 2], True)
    va.add_frame_3D(ax, origin, x_foot_r, y_foot_r, z_foot_r, [1, 0, 0], True)
    va.add_frame_3D(ax, origin, x_thigh_l, y_thigh_l, z_thigh_l, [-1, 0, 4], True)
    va.add_frame_3D(ax, origin, x_shank_l, y_shank_l, z_shank_l, [-1, 0, 2], True)
    va.add_frame_3D(ax, origin, x_foot_l, y_foot_l, z_foot_l, [-1, 0, 0], True)

    # After calibration
    x_pelvis, y_pelvis, z_pelvis    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['pelvis'][i]), seg2sens['pelvis'].T), x_axis, y_axis, z_axis)
    x_thigh_r, y_thigh_r, z_thigh_r = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['thigh_r'][i]), seg2sens['thigh_r'].T), x_axis, y_axis, z_axis)
    x_shank_r, y_shank_r, z_shank_r = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['shank_r'][i]), seg2sens['shank_r'].T), x_axis, y_axis, z_axis)
    x_foot_r, y_foot_r, z_foot_r    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['foot_r'][i]), seg2sens['foot_r'].T), x_axis, y_axis, z_axis)
    x_thigh_l, y_thigh_l, z_thigh_l = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['thigh_l'][i]), seg2sens['thigh_l'].T), x_axis, y_axis, z_axis)
    x_shank_l, y_shank_l, z_shank_l = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['shank_l'][i]), seg2sens['shank_l'].T), x_axis, y_axis, z_axis)
    x_foot_l, y_foot_l, z_foot_l    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(static_orientation_mt['foot_l'][i]), seg2sens['foot_l'].T), x_axis, y_axis, z_axis)

    va.add_frame_3D(ax, origin, x_pelvis, y_pelvis, z_pelvis, [0, 0, 6], False)
    va.add_frame_3D(ax, origin, x_thigh_r, y_thigh_r, z_thigh_r, [1, 0, 4], False)
    va.add_frame_3D(ax, origin, x_shank_r, y_shank_r, z_shank_r, [1, 0, 2], False)
    va.add_frame_3D(ax, origin, x_foot_r, y_foot_r, z_foot_r, [1, 0, 0], False)
    va.add_frame_3D(ax, origin, x_thigh_l, y_thigh_l, z_thigh_l, [-1, 0, 4], False)
    va.add_frame_3D(ax, origin, x_shank_l, y_shank_l, z_shank_l, [-1, 0, 2], False)
    va.add_frame_3D(ax, origin, x_foot_l, y_foot_l, z_foot_l, [-1, 0, 0], False)

    # ax.text(-3, -1, 0, 'frame: ' + str(i))

    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # plt.axis('equal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 7])
    ax.set_aspect('equal', adjustable='box')



# anim = animation.FuncAnimation(fig, update, frames = range(2500, 2700))
anim = animation.FuncAnimation(fig, update)

from matplotlib.animation import PillowWriter
writer = PillowWriter(fps = 100)

anim.save('s' + str(subject) + '_frame_riann_' + selected_setup + '.gif', writer = writer)

# plt.show()

