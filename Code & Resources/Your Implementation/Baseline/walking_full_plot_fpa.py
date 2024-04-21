#read the IMU mtb file
import pandas as pd 
import numpy as np 
import quaternion

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

task_list = ['overground_walking_toe_out']

# Sensor configuration for analysis
sensor_config  = {'pelvis': 'PELV', 
                'foot_r': 'FOOT_L', 'shank_r': 'LLEG_R', 'thigh_r': 'ULEG_R',
                'foot_l': 'FOOT_R', 'shank_l': 'LLEG_L', 'thigh_l': 'ULEG_L',
                'shank_r_mis': 'FARM_R', 'thigh_r_mis': 'UARM_R', 'foot_r_mis': 'HAND_R',}

subject  = 2

# Calibration Step

# Data for calibration
task_static = 'static_pose'
data_static_mt = preprocessing_mt.get_all_data_mt(subject, task_static, sensor_config, stage = 'calibration')
data_static_mt_ = preprocessing_mt.match_data_mt(data_static_mt) # data after matching
task_walking = 'treadmill_walking' # to calibrate thighs, shanks, and feet
data_walking_mt = preprocessing_mt.get_all_data_mt(subject, task_walking, sensor_config, stage = 'calibration')
data_walking_mt_ = preprocessing_mt.match_data_mt(data_walking_mt) # data after matching
task_toe_touching = 'static_toe_touch'
data_toe_touching_mt = preprocessing_mt.get_all_data_mt(subject, task_toe_touching, sensor_config, stage = 'calibration')
data_toe_touching_mt_ = preprocessing_mt.match_data_mt(data_toe_touching_mt) # data after matching
task_static_sitting = 'static_sitting'
data_static_sitting_mt = preprocessing_mt.get_all_data_mt(subject, task_static_sitting, sensor_config, stage = 'calibration')
data_static_sitting_mt_ = preprocessing_mt.match_data_mt(data_static_sitting_mt) # data after matching


walking_period = calibration_mt.get_walking_4_calib(data_walking_mt_['shank_r']['Gyr_Z'].to_numpy())
seg2sens = calibration_mt.sensor_to_segment_mt_cali1(data_static_mt_, data_walking_mt_, walking_period, data_toe_touching_mt_)
seg2sens_2 = calibration_mt.sensor_to_segment_mt_cali3(data_static_mt_, data_toe_touching_mt_, data_static_sitting_mt_)



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
    static_ja_mt_cali1          = ik_mt.get_all_ja_mt(seg2sens, static_orientation_mt)
    static_ja_mt_cali2          = ik_mt.get_all_ja_mt(seg2sens_2, static_orientation_mt)
    static_ja_mt_without_cali   = ik_mt.get_all_ja_mt(no_seg2sens, static_orientation_mt)


    main_orientation_mt = ik_mt.get_imu_orientation_mt(data_main_, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim, params = f_params)
    main_ja_mt          = ik_mt.get_all_ja_mt(seg2sens, main_orientation_mt)

    without_cali_ja_mt = ik_mt.get_all_ja_mt(no_seg2sens, main_orientation_mt)
    cali_2_ja_mt = ik_mt.get_all_ja_mt(seg2sens_2, main_orientation_mt)

    for jk in main_ja_mt.keys():
        offset_cali1         = np.mean(static_ja_mt_cali1[jk])
        offset_cali2         = np.mean(static_ja_mt_cali2[jk])
        offset_without_cali  = np.mean(static_ja_mt_without_cali[jk])
        main_ja_mt[jk] = main_ja_mt[jk] - offset_cali1
        without_cali_ja_mt[jk] = without_cali_ja_mt[jk] - offset_without_cali
        cali_2_ja_mt[jk] = cali_2_ja_mt[jk] - offset_cali2


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
        
        for jk in data_main_.keys():
            data_main_[jk] = 1*data_main_[jk][shifting_id:-1]
        

    start_ja_id = 0
    stop_ja_id  = len(main_ja_mocap['hip_adduction_l'])


    import fpa.FPA_algorithm as fpa
    import fpa.gaitphase as gp
    from fpa.const import EARLY_STANCE, MIDDLE_STANCE, LATE_STANCE, SWING, EULER_INIT_LEN

    rot_our2fpa = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot_our2fpa_2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    # --- FPA --- #
    fpa_obj = fpa.FPA(is_right_foot = True, datarate = constant_mt.MT_SAMPLING_RATE, alpha = 0.8)
    gait_phase_obj = gp.GaitPhase(datarate = constant_mt.MT_SAMPLING_RATE)

    old_gaitphase = gait_phase_obj.gaitphase
    current_gaitphase = None

    # for i in range(gait_Segments['hc_index'][15], gait_Segments['hc_index'][15] + 120):
    fpa_list = []
    for i in range(gait_Segments['hc_index'][3], gait_Segments['hc_index'][-1]):
        acc_x = data_main_['foot_r']['Acc_X'][i]
        acc_y = data_main_['foot_r']['Acc_Y'][i]
        acc_z = data_main_['foot_r']['Acc_Z'][i]
        gyr_x = data_main_['foot_r']['Gyr_X'][i]
        gyr_y = data_main_['foot_r']['Gyr_Y'][i]
        gyr_z = data_main_['foot_r']['Gyr_Z'][i]

        calibrated_acc = np.dot(seg2sens_2['foot_r'], np.array([acc_x, acc_y, acc_z]))
        calibrated_gyr = np.dot(seg2sens_2['foot_r'], np.array([gyr_x, gyr_y, gyr_z]))
        # calibrated_acc = np.dot(seg2sens['foot_r'], np.array([acc_x, acc_y, acc_z]))
        # calibrated_gyr = np.dot(seg2sens['foot_r'], np.array([gyr_x, gyr_y, gyr_z]))

        rotated_acc = np.dot(rot_our2fpa_2, calibrated_acc)
        rotated_gyr = np.dot(rot_our2fpa_2, calibrated_gyr)

        # rotated_acc = np.dot(rot_our2fpa, np.array([acc_x, acc_y, acc_z]))
        # rotated_gyr = np.dot(rot_our2fpa, np.array([gyr_x, gyr_y, gyr_z]))

        data = {'AccelX': rotated_acc[0], 'AccelY': rotated_acc[1], 'AccelZ': rotated_acc[2],
                'GyroX': rotated_gyr[0], 'GyroY': rotated_gyr[1], 'GyroZ': rotated_gyr[2]}


        # data = {'AccelX': data_main_['foot_r']['Acc_Y'][i], 'AccelY': (-1) * data_main_['foot_r']['Acc_X'][i], 'AccelZ': data_main_['foot_r']['Acc_Z'][i],
        #         'GyroX': data_main_['foot_r']['Gyr_X'][i], 'GyroY': data_main_['foot_r']['Gyr_Y'][i], 'GyroZ': data_main_['foot_r']['Gyr_Z'][i]}
        
        # turn rad to degrees
        data['GyroX'] = np.rad2deg(data['GyroX'])
        data['GyroY'] = np.rad2deg(data['GyroY'])
        data['GyroZ'] = np.rad2deg(data['GyroZ'])

        gait_phase_obj.update_gaitphase(data)
        current_gaitphase = gait_phase_obj.gaitphase
        # if current_gaitphase == MIDDLE_STANCE:
        fpa_obj.update_FPA(data, gait_phase_obj.gaitphase_old, gait_phase_obj.gaitphase)
        print(fpa_obj.FPA_this_step)
        # if fpa_obj.FPA_this_step > 0:
        #     fpa_list.append(fpa_obj.FPA_this_step)
        fpa_list.append(fpa_obj.FPA_this_step)

    #remove outliers
    fpa_list = np.array(fpa_list)
    upper_bound = np.mean(fpa_list) + 1*np.std(fpa_list)
    lower_bound = np.mean(fpa_list) - 1*np.std(fpa_list)
    fpa_list = fpa_list[(fpa_list < upper_bound) & (fpa_list > lower_bound)]

    #flip the sign
    # fpa_list = -1*fpa_list
    
    #save the fpa_list to a csv file
    fpa_list = pd.DataFrame(fpa_list)
    fpa_list.to_csv('fpa_list_cali2_toe_out.csv')




        # fpa_obj.update_FPA(data, gait_phase_obj.gaitphase_old, gait_phase_obj.gaitphase)

    #plot the FPA
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(fpa_list)
    ax.set_xlabel('Time')
    ax.set_ylabel('FPA $(^o)$')
    ax.set_title('Foot Progression Angle')
    plt.show()


    average_fpa = np.mean(fpa_list)
    print('Average FPA: ', average_fpa)


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


    # #draw segmented joint angles
    # segment_id = gait_Segments['hc_index']






    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows = 3, ncols = 3, sharex = True)
    # fig.set_size_inches(12, 7)
    # #set up the 2d dictionary
    # kinematic_angle = {
    #     'hip': {
    #         'adduction': {},
    #         'rotation': {},
    #         'flexion': {}
    #     },
    #     'knee': {
    #         'adduction': {},
    #         'rotation': {},
    #         'flexion': {}
    #     },
    #     'ankle': {
    #         'adduction': {},
    #         'rotation': {},
    #         'flexion': {}
    #     }

    # }

    # row = 0
    # col = 0
    # for joint_id in ('hip', 'knee', 'ankle'):
    #     for angle_id in ('adduction', 'rotation', 'flexion'):
    #         kinematic_angle = joint_id + '_' + angle_id + '_r'
    #         s_without_cali = []
    #         s_cali_1 = []
    #         s_mocap = []
    #         s_cali_2 = []
    #         for i in range(len(segment_id) - 1):
    #             N = segment_id[i+1] - segment_id[i]
    #             if N > 200:
    #                 continue
    #             # s_os.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_os_ja[segment_id[i]:segment_id[i+1]]))
    #             s_cali_1.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), main_ja_mt[kinematic_angle][segment_id[i]:segment_id[i+1]]))
    #             s_mocap.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), main_ja_mocap[kinematic_angle][segment_id[i]:segment_id[i+1]]))
    #             s_without_cali.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), without_cali_ja_mt[kinematic_angle][segment_id[i]:segment_id[i+1]]))
    #             s_cali_2.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), cali_2_ja_mt[kinematic_angle][segment_id[i]:segment_id[i+1]]))
    #         plt.rcParams.update({'font.size': 10})
    #         ax[row, col].plot(100*np.linspace(0, 1, 100), np.mean(s_mocap, axis = 0), linewidth = 1.5, linestyle = '--', color = '#000000', label = 'Motion capture')
    #         ax[row, col].plot(100*np.linspace(0, 1, 100), np.mean(s_cali_1, axis = 0), linewidth = 1.5, color = '#ad343e', label = 'IMU Calibration 1')
    #         ax[row, col].plot(100*np.linspace(0, 1, 100), np.mean(s_without_cali, axis = 0), linewidth = 1.5, linestyle =':' , color = '#134074', label = 'IMU Without Calibration')
    #         ax[row, col].plot(100*np.linspace(0, 1, 100), np.mean(s_cali_2, axis = 0), linewidth = 1.5, linestyle = '-.', color = '#f2af29', label = 'IMU Calibration 2')
    #         ax[row, col].fill_between(100*np.linspace(0, 1, 100), (np.mean(s_mocap, axis = 0)-2*np.std(s_mocap, axis = 0)), (np.mean(s_mocap, axis = 0)+2*np.std(s_mocap, axis = 0)), color = '#000000', alpha=.3)
    #         ax[row, col].fill_between(100*np.linspace(0, 1, 100), (np.mean(s_cali_1, axis = 0)-2*np.std(s_cali_1, axis = 0)), (np.mean(s_cali_1, axis = 0)+2*np.std(s_cali_1, axis = 0)), color = '#ad343e', alpha=.3)
    #         ax[row, col].fill_between(100*np.linspace(0, 1, 100), (np.mean(s_without_cali, axis = 0)-2*np.std(s_without_cali, axis = 0)), (np.mean(s_without_cali, axis = 0)+2*np.std(s_without_cali, axis = 0)), color = '#134074', alpha=.3)
    #         ax[row, col].fill_between(100*np.linspace(0, 1, 100), (np.mean(s_cali_2, axis = 0)-2*np.std(s_cali_2, axis = 0)), (np.mean(s_cali_2, axis = 0)+2*np.std(s_cali_2, axis = 0)), color = '#f2af29', alpha=.3)
    #         # ax[row, col].set_ylim([-30, 120])
    #         # ax.set_ylim([-40, 60])
    #         # ax.set_ylim([-40, 40])
    #         ax[row, col].set_xlim([0, 100])
    #         ax[row, col].set_xlabel('Gait cycle')
    #         ax[row, col].set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    #         # ax[row, col].set_ylabel('Angle $(^o)$')
    #         ax[row, col].spines['left'].set_position(('outward', 8))
    #         ax[row, col].spines['bottom'].set_position(('outward', 5))
    #         ax[row, col].spines['top'].set_visible(False)
    #         ax[row, col].spines['right'].set_visible(False)
    #         # ax[row, col].legend(loc = 'upper left', frameon = False, prop={'size': 10}, ncol = 1)
    #         row += 1
    #         if row == 3:
    #             row = 0
    #     col += 1
    # #add only one legend to the overall plot group
    # ax[0, 2].legend(loc = 'upper left', frameon = False, prop={'size': 10}, ncol = 1, bbox_to_anchor=(1.01, 1))
    # # add title to each row and col
    # ax[0, 0].set_title('Hip')
    # ax[0, 1].set_title('Knee')
    # ax[0, 2].set_title('Ankle')
    # ax[0, 0].set_ylabel('Adduction Angle $(^o)$')
    # ax[1, 0].set_ylabel('Rotation Angle $(^o)$')
    # ax[2, 0].set_ylabel('Flexion Angle $(^o)$')

    # #remove the white space on the left side
    # plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)




    # fig.suptitle('Joint Angles Comparison for Overground Walking')
    # plt.show()
    # # plt.savefig("eni_slide_legend.png", bbox_inches='tight')
    # print('<-- Done')



    # # # --- Save the results --- #
    # mocap_ja_storage = []
    # cali_1_ja_storage = []
    # without_cali_ja_storage = []
    # cali_2_ja_storage = []
    # for jk in main_ja_mocap.keys():
    #     mocap_ja_storage.append(main_ja_mocap[jk])
    #     cali_1_ja_storage.append(main_ja_mt[jk])
    #     without_cali_ja_storage.append(without_cali_ja_mt[jk])
    #     cali_2_ja_storage.append(cali_2_ja_mt[jk])
    # mocap_ja_storage = np.array(mocap_ja_storage)
    # cali_1_ja_storage = np.array(cali_1_ja_storage)
    # without_cali_ja_storage = np.array(without_cali_ja_storage)
    # cali_2_ja_storage = np.array(cali_2_ja_storage)
    # mocap_ja_storage = pd.DataFrame(mocap_ja_storage)
    # cali_1_ja_storage = pd.DataFrame(cali_1_ja_storage)
    # without_cali_ja_storage = pd.DataFrame(without_cali_ja_storage)
    # cali_2_ja_storage = pd.DataFrame(cali_2_ja_storage)
    # mocap_ja_storage.to_csv('Milestone_4/s' + str(subject) + '_' + selected_task + '_mocap_results.csv')
    # cali_1_ja_storage.to_csv('Milestone_4/s' + str(subject) + '_' + selected_task + '_cali_1_results.csv')
    # without_cali_ja_storage.to_csv('Milestone_4/s' + str(subject) + '_' + selected_task + '_without_cali_results.csv')
    # cali_2_ja_storage.to_csv('Milestone_4/s' + str(subject) + '_' + selected_task + '_cali_2_results.csv')
