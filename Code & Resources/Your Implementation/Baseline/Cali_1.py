#read the IMU mtb file
import pandas as pd 
import numpy as np 
import quaternion
import matplotlib.pyplot as plt

from utils import va, synchronization, common, constant_common
from utils.mt import constant_mt, preprocessing_mt, calibration_mt, ik_mt
from utils.mocap import constant_mocap, preprocessing_mocap, ik_mocap, gait_params_mocap



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

# task_list = ['hip_adduction', 'knee_flexion_ex', 'leg_swing', 'overground_jogging',
#              'overground_walking', 'overground_walking_toe_in', 'overground_walking_toe_out',
#              'sts_jump', 'vertical_jump']

task_list = ['overground_walking']

# Sensor configuration for analysis
sensor_config  = {'pelvis': 'PELV', 
                'foot_r': 'FOOT_R', 'shank_r': 'LLEG_R', 'thigh_r': 'ULEG_R',
                'foot_l': 'FOOT_L', 'shank_l': 'LLEG_L', 'thigh_l': 'ULEG_L',
                'shank_r_mis': 'FARM_R', 'thigh_r_mis': 'UARM_R', 'foot_r_mis': 'HAND_R'}

subject  = 2

# Calibration Step

# Data for calibration
task_static = 'static_pose'
data_static_mt = preprocessing_mt.get_all_data_mt(subject, task_static, sensor_config, stage = 'calibration')
data_static_mt_ = preprocessing_mt.match_data_mt(data_static_mt) # data after matching
task_walking = 'treadmill_walking' # to calibrate thighs, shanks, and feet
data_walking_mt = preprocessing_mt.get_all_data_mt(subject, task_walking, sensor_config, stage = 'calibration')
task_toe_touching = 'static_toe_touch'
data_toe_touching_mt = preprocessing_mt.get_all_data_mt(subject, task_toe_touching, sensor_config, stage = 'calibration')
task_static_sitting = 'static_sitting'
data_static_sitting_mt = preprocessing_mt.get_all_data_mt(subject, task_static_sitting, sensor_config, stage = 'calibration')

walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Z'].to_numpy())
seg2sens = calibration_mt.sensor_to_segment_mt_cali1(data_static_mt, data_walking_mt, walking_period, data_toe_touching_mt)
seg2sens_2 = calibration_mt.sensor_to_segment_mt_cali3(data_static_mt, data_toe_touching_mt, data_static_sitting_mt)



for selected_task in task_list:
    # selected_task  = 'vertical_jump'
    print('*** Subject ' + str(subject))
    print('*** Selected Task ' + selected_task)


    # Data for analysis
    data_main  = preprocessing_mt.get_all_data_mt(subject, selected_task, sensor_config, stage = 'evaluation')
    data_main_ = preprocessing_mt.match_data_mt(data_main) # data after matching




    # # Calibration
    no_seg2sens = {}
    for sensor_id in sensor_config.keys():
        no_seg2sens[sensor_id] = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])

    print(seg2sens)

    # # TODO: get joint angles during task

    static_orientation_mt = ik_mt.get_imu_orientation_mt(data_static_mt_, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim, params = f_params)
    static_ja_mt          = ik_mt.get_all_ja_mt(seg2sens, static_orientation_mt)


    main_orientation_mt = ik_mt.get_imu_orientation_mt(data_main_, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim, params = f_params)
    main_ja_mt          = ik_mt.get_all_ja_mt(seg2sens, main_orientation_mt)

    without_cali_ja_mt = ik_mt.get_all_ja_mt(no_seg2sens, main_orientation_mt)
    cali_2_ja_mt = ik_mt.get_all_ja_mt(seg2sens_2, main_orientation_mt)

    for jk in main_ja_mt.keys():
        offset         = np.mean(static_ja_mt[jk])
        main_ja_mt[jk] = main_ja_mt[jk] - offset
        without_cali_ja_mt[jk] = without_cali_ja_mt[jk] - offset
        cali_2_ja_mt[jk] = cali_2_ja_mt[jk] - offset


    # # --- Mocap data --- #

    # Data for analysis
    data_static_mocap     = preprocessing_mocap.get_data_mocap(subject, task_static, stage="full")
    data_static_mocap     = data_static_mocap.interpolate(method = 'cubic')
    data_static_mocap     = data_static_mocap.fillna(value = 999)
    data_static_mocap_avg = preprocessing_mocap.get_avg_data(data_static_mocap)



    data_main_mocap = preprocessing_mocap.get_data_mocap(subject, selected_task, stage = "evaluation")
    data_main_mocap = data_main_mocap.interpolate(method = 'cubic')
    data_main_mocap = data_main_mocap.fillna(value = 999)
    data_main_mocap = preprocessing_mocap.lowpass_filter_mocap(data_main_mocap, constant_mocap.MOCAP_SAMPLING_RATE,
                                                            constant_mocap.FILTER_CUTOFF_MOCAP,
                                                            constant_mocap.FILTER_ORDER) # filter
    data_main_mocap = preprocessing_mocap.resample_mocap(data_main_mocap, constant_mt.MT_SAMPLING_RATE) # downsample

    gait_Segments = gait_params_mocap.get_hc_n_to_mocap(data_main_mocap['RCAL Y'], data_main_mocap['RMT1 Y'], 'walking', fs = constant_mt.MT_SAMPLING_RATE, remove = 600, vis = False)

    # Get orientation
    # main_orientation_mocap       = ik_mocap.get_orientation_mocap(data_main_mocap, cluster_use = True)
    # cal_orientation_mocap        = ik_mocap.calibration_mocap(cluster_use = False)

    # # Get joint angles
    # main_ja_mocap   = ik_mocap.get_all_ja_mocap(cal_orientation_mocap, main_orientation_mocap, cluster_use = True)


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



    # # --- Sync mocap and IMU --- #
    if selected_task in ['lat_step', 'step_up_down', 'drop_jump', 'cmj', 'squat', 'step_n_hold', 'sls', 'sts']:
        iters = 600 # 800 previously
    else:
        iters = 1500
    main_ja_mocap, main_ja_mt, first_start, shifting_id = synchronization.sync_ja_mocap_mt(main_ja_mocap, data_main_mocap, main_ja_mt, data_main_, iters = iters)


    if first_start == 'mocap':
        gait_Segments['hc_index'] -= shifting_id
        gait_Segments['to_index'] -= shifting_id
    else:
        for jk in without_cali_ja_mt.keys():
            without_cali_ja_mt[jk]   = 1*without_cali_ja_mt[jk][shifting_id:-1]
            cali_2_ja_mt[jk]         = 1*cali_2_ja_mt[jk][shifting_id:-1]

    start_ja_id = 0
    stop_ja_id  = len(main_ja_mocap['hip_adduction_l'])


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows = 9, ncols = 2, sharex = True)
    # ax[0, 0].plot(main_ja_mocap['hip_adduction_l'])
    # ax[1, 0].plot(main_ja_mocap['hip_rotation_l'])
    # ax[2, 0].plot(main_ja_mocap['hip_flexion_l'])
    # ax[3, 0].plot(main_ja_mocap['knee_adduction_l'])
    # ax[4, 0].plot(main_ja_mocap['knee_rotation_l'])
    # ax[5, 0].plot(main_ja_mocap['knee_flexion_l'])
    # ax[6, 0].plot(main_ja_mocap['ankle_adduction_l'])
    # ax[7, 0].plot(main_ja_mocap['ankle_rotation_l'])
    # ax[8, 0].plot(main_ja_mocap['ankle_flexion_l'])


    # ax[0, 1].plot(main_ja_mocap['hip_adduction_r'])
    # ax[1, 1].plot(main_ja_mocap['hip_rotation_r'])
    # ax[2, 1].plot(main_ja_mocap['hip_flexion_r'])
    # ax[3, 1].plot(main_ja_mocap['knee_adduction_r'])
    # ax[4, 1].plot(main_ja_mocap['knee_rotation_r'])
    # ax[5, 1].plot(main_ja_mocap['knee_flexion_r'])
    # ax[6, 1].plot(main_ja_mocap['ankle_adduction_r'])
    # ax[7, 1].plot(main_ja_mocap['ankle_rotation_r'])
    # ax[8, 1].plot(main_ja_mocap['ankle_flexion_r'])


    # ax[0, 0].plot(main_ja_mt['hip_adduction_l'])
    # ax[1, 0].plot(main_ja_mt['hip_rotation_l'])
    # ax[2, 0].plot(main_ja_mt['hip_flexion_l'])
    # ax[3, 0].plot(main_ja_mt['knee_adduction_l'])
    # ax[4, 0].plot(main_ja_mt['knee_rotation_l'])
    # ax[5, 0].plot(main_ja_mt['knee_flexion_l'])
    # ax[6, 0].plot(main_ja_mt['ankle_adduction_l'])
    # ax[7, 0].plot(main_ja_mt['ankle_rotation_l'])
    # ax[8, 0].plot(main_ja_mt['ankle_flexion_l'])



    # ax[0, 1].plot(main_ja_mt['hip_adduction_r'])
    # ax[1, 1].plot(main_ja_mt['hip_rotation_r'])
    # ax[2, 1].plot(main_ja_mt['hip_flexion_r'])
    # ax[3, 1].plot(main_ja_mt['knee_adduction_r'])
    # ax[4, 1].plot(main_ja_mt['knee_rotation_r'])
    # ax[5, 1].plot(main_ja_mt['knee_flexion_r'])
    # ax[6, 1].plot(main_ja_mt['ankle_adduction_r'])
    # ax[7, 1].plot(main_ja_mt['ankle_rotation_r'])
    # ax[8, 1].plot(main_ja_mt['ankle_flexion_r'])

    # #add labels
    # ax[0, 0].set_ylabel('hip_adduction_l')
    # ax[1, 0].set_ylabel('hip_rotation_l')
    # ax[2, 0].set_ylabel('hip_flexion_l')
    # ax[3, 0].set_ylabel('knee_adduction_l')
    # ax[4, 0].set_ylabel('knee_rotation_l')
    # ax[5, 0].set_ylabel('knee_flexion_l')
    # ax[6, 0].set_ylabel('ankle_adduction_l')
    # ax[7, 0].set_ylabel('ankle_rotation_l')
    # ax[8, 0].set_ylabel('ankle_flexion_l')


    # ax[0, 1].set_ylabel('hip_adduction_r')
    # ax[1, 1].set_ylabel('hip_rotation_r')
    # ax[2, 1].set_ylabel('hip_flexion_r')
    # ax[3, 1].set_ylabel('knee_adduction_r')
    # ax[4, 1].set_ylabel('knee_rotation_r')
    # ax[5, 1].set_ylabel('knee_flexion_r')
    # ax[6, 1].set_ylabel('ankle_adduction_r')
    # ax[7, 1].set_ylabel('ankle_rotation_r')
    # ax[8, 1].set_ylabel('ankle_flexion_r')

    # #add legend
    # ax[0, 0].legend(['Mocap', 'IMU'])
    # ax[1, 0].legend(['Mocap', 'IMU'])
    # ax[2, 0].legend(['Mocap', 'IMU'])
    # ax[3, 0].legend(['Mocap', 'IMU'])
    # ax[4, 0].legend(['Mocap', 'IMU'])
    # ax[5, 0].legend(['Mocap', 'IMU'])
    # ax[6, 0].legend(['Mocap', 'IMU'])
    # ax[7, 0].legend(['Mocap', 'IMU'])
    # ax[8, 0].legend(['Mocap', 'IMU'])

    # ax[0, 1].legend(['Mocap', 'IMU'])
    # ax[1, 1].legend(['Mocap', 'IMU'])
    # ax[2, 1].legend(['Mocap', 'IMU'])
    # ax[3, 1].legend(['Mocap', 'IMU'])
    # ax[4, 1].legend(['Mocap', 'IMU'])
    # ax[5, 1].legend(['Mocap', 'IMU'])
    # ax[6, 1].legend(['Mocap', 'IMU'])
    # ax[7, 1].legend(['Mocap', 'IMU'])
    # ax[8, 1].legend(['Mocap', 'IMU'])

    # fig.suptitle(selected_task + ' Joint Angles Comparison')
    # plt.show()


    #draw segmented joint angles
    segment_id = gait_Segments['hc_index']
    # gait_length = []
    # for i in range(len(segment_id) - 1):
    #     gait_length.append(segment_id[i+1] - segment_id[i])
    
    # # plot the gait cycle length as dots
    # plt.plot(gait_length, 'o')
    # plt.show()





    s_without_cali = []
    s_cali_1 = []
    s_mocap = []
    s_cali_2 = []
    for i in range(len(segment_id) - 1):
        N = segment_id[i+1] - segment_id[i]
        if N > 200:
            continue
        # s_os.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_os_ja[segment_id[i]:segment_id[i+1]]))
        s_cali_1.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), main_ja_mt['knee_flexion_r'][segment_id[i]:segment_id[i+1]]))
        s_mocap.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), main_ja_mocap['knee_flexion_r'][segment_id[i]:segment_id[i+1]]))
        s_without_cali.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), without_cali_ja_mt['knee_flexion_r'][segment_id[i]:segment_id[i+1]]))
        s_cali_2.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), cali_2_ja_mt['knee_flexion_r'][segment_id[i]:segment_id[i+1]]))
    # df_mocap = pd.DataFrame(np.array(s_mocap))
    # df_xsens = pd.DataFrame(np.array(s_xsens))
    # df_mocap.to_csv('s4_mocap_ankle.csv')
    # df_xsens.to_csv('s4_xsens_ankle_unc_misp.csv')
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(100*np.linspace(0, 1, 100), np.mean(s_mocap, axis = 0), linewidth = 2.5, linestyle = '--', color = '#000000', label = 'Motion capture')
    ax.plot(100*np.linspace(0, 1, 100), np.mean(s_cali_1, axis = 0), linewidth = 2.5, color = '#ad343e', label = 'IMU Calibration 1')
    ax.plot(100*np.linspace(0, 1, 100), np.mean(s_without_cali, axis = 0), linewidth = 2.5, linestyle =':' , color = '#134074', label = 'IMU Without Calibration')
    ax.plot(100*np.linspace(0, 1, 100), np.mean(s_cali_2, axis = 0), linewidth = 2.5, linestyle = '-.', color = '#f2af29', label = 'IMU Calibration 2')
    ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_mocap, axis = 0)-2*np.std(s_mocap, axis = 0)), (np.mean(s_mocap, axis = 0)+2*np.std(s_mocap, axis = 0)), color = '#000000', alpha=.3)
    ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_cali_1, axis = 0)-2*np.std(s_cali_1, axis = 0)), (np.mean(s_cali_1, axis = 0)+2*np.std(s_cali_1, axis = 0)), color = '#ad343e', alpha=.3)
    ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_without_cali, axis = 0)-2*np.std(s_without_cali, axis = 0)), (np.mean(s_without_cali, axis = 0)+2*np.std(s_without_cali, axis = 0)), color = '#134074', alpha=.3)
    ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_cali_2, axis = 0)-2*np.std(s_cali_2, axis = 0)), (np.mean(s_cali_2, axis = 0)+2*np.std(s_cali_2, axis = 0)), color = '#f2af29', alpha=.3)
    ax.set_ylim([0, 120])
    # ax.set_ylim([-40, 60])
    # ax.set_ylim([-40, 40])
    ax.set_xlim([0, 100])
    ax.set_xlabel('Gait cycle')
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_ylabel('Angle $(^o)$')
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = 'upper left', frameon = False, prop={'size': 18}, ncol = 1)
    plt.show()
    # plt.savefig("eni_slide_legend.png", bbox_inches='tight')
    print('<-- Done')



    # #save the main_ja_mocap as csv file
    # mocap_ja_storage = []
    # for jk in main_ja_mocap.keys():
    #     mocap_ja_storage.append(main_ja_mocap[jk])

    # mocap_ja_storage = np.array(mocap_ja_storage)
    # # mocap_ja_storage = mocap_ja_storage.reshape((18, -1))
    # mocap_ja_storage = pd.DataFrame(mocap_ja_storage)
    # mocap_ja_storage.to_csv('s' + str(subject) + '_' + selected_task + '_mocap_results.csv')

    # #save the main_ja_mt as csv file
    # mt_ja_storage = []
    # for jk in main_ja_mt.keys():
    #     mt_ja_storage.append(main_ja_mt[jk])

    # mt_ja_storage = np.array(mt_ja_storage)
    # # mt_ja_storage = mt_ja_storage.reshape((18, -1))
    # mt_ja_storage = pd.DataFrame(mt_ja_storage)
    # mt_ja_storage.to_csv('s' + str(subject) + '_' + selected_task + '_mt_results.csv')


    # temp_ja_storage = []

    # for jk in main_ja_mt.keys():
    #     print(jk)
    #     temp_rmse = common.get_rmse(main_ja_mocap[jk][start_ja_id:stop_ja_id], main_ja_mt[jk][start_ja_id:stop_ja_id])
    #     temp_ja_storage.append(temp_rmse)

    # temp_ja_storage = np.array(temp_ja_storage)
    # temp_ja_storage = temp_ja_storage.reshape((1, len(temp_ja_storage)))
    # temp_ja_storage = pd.DataFrame(temp_ja_storage)
    # temp_ja_storage.to_csv('s' + str(subject) + '_' + selected_task + '_results.csv')


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, writers
# from matplotlib import animation

# fig = plt.figure(figsize = (8, 8))
# ax = fig.add_subplot(projection='3d')

# origin = np.array([[0], [0], [0]])
# x_axis = np.array([[1], [0], [0]])
# y_axis = np.array([[0], [1], [0]])
# z_axis = np.array([[0], [0], [1]])

# def update(i):
#     ax.clear()

#     # Before calibration
#     x_pelvis, y_pelvis, z_pelvis    = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['pelvis'][i]), x_axis, y_axis, z_axis)
#     x_thigh_r, y_thigh_r, z_thigh_r = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['thigh_r'][i]), x_axis, y_axis, z_axis)
#     x_shank_r, y_shank_r, z_shank_r = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['shank_r'][i]), x_axis, y_axis, z_axis)
#     x_foot_r, y_foot_r, z_foot_r    = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['foot_r'][i]), x_axis, y_axis, z_axis)
#     x_thigh_l, y_thigh_l, z_thigh_l = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['thigh_l'][i]), x_axis, y_axis, z_axis)
#     x_shank_l, y_shank_l, z_shank_l = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['shank_l'][i]), x_axis, y_axis, z_axis)
#     x_foot_l, y_foot_l, z_foot_l    = va.rotate_frame(quaternion.as_rotation_matrix(main_orientation_mt['foot_l'][i]), x_axis, y_axis, z_axis)

#     va.add_frame_3D(ax, origin, x_pelvis, y_pelvis, z_pelvis, [0, 0, 6], True)
#     va.add_frame_3D(ax, origin, x_thigh_r, y_thigh_r, z_thigh_r, [1, 0, 4], True)
#     va.add_frame_3D(ax, origin, x_shank_r, y_shank_r, z_shank_r, [1, 0, 2], True)
#     va.add_frame_3D(ax, origin, x_foot_r, y_foot_r, z_foot_r, [1, 0, 0], True)
#     va.add_frame_3D(ax, origin, x_thigh_l, y_thigh_l, z_thigh_l, [-1, 0, 4], True)
#     va.add_frame_3D(ax, origin, x_shank_l, y_shank_l, z_shank_l, [-1, 0, 2], True)
#     va.add_frame_3D(ax, origin, x_foot_l, y_foot_l, z_foot_l, [-1, 0, 0], True)

#     # After calibration
#     x_pelvis, y_pelvis, z_pelvis    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['pelvis'][i]), seg2sens_2['pelvis'].T), x_axis, y_axis, z_axis)
#     x_thigh_r, y_thigh_r, z_thigh_r = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['thigh_r'][i]), seg2sens_2['thigh_r'].T), x_axis, y_axis, z_axis)
#     x_shank_r, y_shank_r, z_shank_r = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['shank_r'][i]), seg2sens_2['shank_r'].T), x_axis, y_axis, z_axis)
#     x_foot_r, y_foot_r, z_foot_r    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['foot_r'][i]), seg2sens_2['foot_r'].T), x_axis, y_axis, z_axis)
#     x_thigh_l, y_thigh_l, z_thigh_l = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['thigh_l'][i]), seg2sens_2['thigh_l'].T), x_axis, y_axis, z_axis)
#     x_shank_l, y_shank_l, z_shank_l = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['shank_l'][i]), seg2sens_2['shank_l'].T), x_axis, y_axis, z_axis)
#     x_foot_l, y_foot_l, z_foot_l    = va.rotate_frame(np.matmul(quaternion.as_rotation_matrix(main_orientation_mt['foot_l'][i]), seg2sens_2['foot_l'].T), x_axis, y_axis, z_axis)

#     va.add_frame_3D(ax, origin, x_pelvis, y_pelvis, z_pelvis, [0, 0, 6], False)
#     va.add_frame_3D(ax, origin, x_thigh_r, y_thigh_r, z_thigh_r, [1, 0, 4], False)
#     va.add_frame_3D(ax, origin, x_shank_r, y_shank_r, z_shank_r, [1, 0, 2], False)
#     va.add_frame_3D(ax, origin, x_foot_r, y_foot_r, z_foot_r, [1, 0, 0], False)
#     va.add_frame_3D(ax, origin, x_thigh_l, y_thigh_l, z_thigh_l, [-1, 0, 4], False)
#     va.add_frame_3D(ax, origin, x_shank_l, y_shank_l, z_shank_l, [-1, 0, 2], False)
#     va.add_frame_3D(ax, origin, x_foot_l, y_foot_l, z_foot_l, [-1, 0, 0], False)

#     # ax.text(-3, -1, 0, 'frame: ' + str(i))

#     ax.grid(False)
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

#     # plt.axis('equal')
#     ax.set_xlim([-2, 2])
#     ax.set_ylim([-2, 2])
#     ax.set_zlim([0, 7])
#     ax.set_aspect('equal', adjustable='box')



# anim = animation.FuncAnimation(fig, update, frames = range(0, 300))
# # anim = animation.FuncAnimation(fig, update)

# from matplotlib.animation import PillowWriter
# writer = PillowWriter(fps = 100)

# anim.save('s' + str(subject) + '_frame_riann_walking_cali2_1' + '.gif', writer = writer)