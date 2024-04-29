# name: calibration_mt.py
# description: sensor-to-body alignment for IMUs
# author: Vu Phan
# date: 2024/01/27


import numpy as np 

from numpy.linalg import norm, inv 
from sklearn.decomposition import PCA 
from tqdm import tqdm 

import sys, os 
sys.path.append('/path/to/IMU_Kinematics_Comparison_v2')

from utils.mt import constant_mt
from utils.mt import segmentation_mt


# --- Get PCA axis --- #
def get_pc1_ax_mt(data):
    ''' Get the rotation axis during walking (for thighs/shanks/feet) or squat (for pelvis) using PCA

    Args:
        + data (pd.DataFrame): walking data of a thigh/shank sensor or squat data of the pelvis sensor

    Returns:
        + pc1_ax (np.array): the first principal component of data
    '''
    data = data - np.mean(data, axis = 0)
    pca  = PCA(n_components = 3)
    pca.fit(data)

    pc1_ax = 1*pca.components_[0]

    return pc1_ax


# --- Sensor-to-segment alignment (calibration) --- #
# Get walking period for calibration
def get_walking_4_calib(shank_walking_gyr_r):
    ''' Get walking period for calibration 

    Args:
        + shank_walking_gyr_r (np.array): gyroscope data of the right shank during walking
    
    Returns:
        + period (list of int): period of walking for calibration
    '''
    gait_events = segmentation_mt.detect_gait_events(shank_walking_gyr_r)
    
    period = [gait_events['ms_index'][10], gait_events['ms_index'][18]]

    return period

# Get squat period for calibration
def get_squat_4_calib(pelvis_squat_gyr):
    ''' Get squat period for calibration

    Args:
        + pelvis_squat_gyr (np.array): gyroscope data of the pelvis during squat

    Returns:
        + period (list of int): period of squat for calibration
    ''' 
    rep_events = segmentation_mt.detect_exercise_rep(pelvis_squat_gyr)

    period = [rep_events['start_index'], rep_events['stop_index']]

    return period

# Calibration
def sensor_to_segment_mt(data_static, data_walking, walking_period, data_squat, squat_period, dir):
    ''' Obtain transformation from segment-to-sensor

    Args:
        + data_static (dict of pd.DataFrame): static data for the vertical axis
        + data_walking (dict of pd.DataFrame): walking data for thigh/shank/foot rotational axis
        + data_squat (dict of pd.DataFrame): squat data for pelvis rotational axis
        + dir (str): direction of attachment, e.g., mid, high, low, or front

    Returns:
        + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
    '''
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            squat_gyr = 1*data_squat[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            squat_gyr = squat_gyr[squat_period[0]:squat_period[1], :]
            pc1_ax    = get_pc1_ax_mt(squat_gyr)

            if pc1_ax[1] > 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)

        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        else:
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if dir == 'front':
                if pc1_ax[1] < 0:
                    pc1_ax = (-1)*pc1_ax
                
                vx = np.cross(fy, pc1_ax)
                fx = vx/norm(vx)

                vz = np.cross(fx, fy)
                fz = vz/norm(vz)

            else:
                if pc1_ax[-1] < 0:
                    pc1_ax = (-1)*pc1_ax
                
                if side == 'r':
                    vx = np.cross(fy, pc1_ax)
                else:
                    vx = np.cross(pc1_ax, fy)
                
                fx = vx/norm(vx)

                vz = np.cross(fx, fy)
                fz = vz/norm(vz)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens


# Calibration
def sensor_to_segment_mt_cali1(data_static, data_walking, walking_period, data_toe_touching):
    ''' Obtain transformation from segment-to-sensor

    Args:
        + data_static (dict of pd.DataFrame): static data for the vertical axis
        + data_walking (dict of pd.DataFrame): walking data for thigh/shank/foot rotational axis
        + data_squat (dict of pd.DataFrame): squat data for pelvis rotational axis
        + dir (str): direction of attachment, e.g., mid, high, low, or front

    Returns:
        + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
    '''
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            static_toe_touching_acc = 1*data_toe_touching[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
            # vx = np.mean(static_toe_touching_acc, axis = 0)

            vy_1 = np.mean(static_toe_touching_acc, axis = 0)
            fy_1 = vy_1/norm(vy_1)

            vz = np.cross(fy, fy_1)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)

        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        else:
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)


            if pc1_ax[-1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            if side == 'r':
                vx = np.cross(fy, pc1_ax)
            else:
                vx = np.cross(pc1_ax, fy)
            
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens


def sensor_to_segment_mt_cali2(data_static, data_toe_touching, data_leg_swing_left, data_leg_swing_right):
    ''' Obtain transformation from segment-to-sensor

    Args:
        + data_static (dict of pd.DataFrame): static data for the vertical axis
        + data_walking (dict of pd.DataFrame): walking data for thigh/shank/foot rotational axis
        + data_squat (dict of pd.DataFrame): squat data for pelvis rotational axis
        + dir (str): direction of attachment, e.g., mid, high, low, or front

    Returns:
        + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
    '''
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            static_toe_touching_acc = 1*data_toe_touching[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
            vx = np.mean(static_toe_touching_acc, axis = 0)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)


        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            if side == 'r':
                leg_swing_data = data_leg_swing_right
            else:
                leg_swing_data = data_leg_swing_left
            
            leg_swing_gyr = 1*leg_swing_data[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            pc1_ax      = get_pc1_ax_mt(leg_swing_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        else:
            if side == 'r':
                leg_swing_data = data_leg_swing_right
            else:
                leg_swing_data = data_leg_swing_left
            
            leg_swing_gyr = 1*leg_swing_data[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            pc1_ax      = get_pc1_ax_mt(leg_swing_gyr)


            if pc1_ax[-1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            if side == 'r':
                vx = np.cross(fy, pc1_ax)
            else:
                vx = np.cross(pc1_ax, fy)
            
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens

def sensor_to_segment_mt_cali3(data_static, data_toe_touching, data_static_sitting):
    
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            static_toe_touching_acc = 1*data_toe_touching[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()

            vy_1 = np.mean(static_toe_touching_acc, axis = 0)
            fy_1 = vy_1/norm(vy_1)

            vz = np.cross(fy, fy_1)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)

        else:
            static_sitting_acc = 1*data_static_sitting[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
            vy_1 = np.mean(static_sitting_acc, axis = 0)
            fy_1 = vy_1/norm(vy_1)

            vz = np.cross(fy_1, fy)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens


def sensor_to_segment_mt_cali1_flip_axis(data_static, data_walking, walking_period, data_toe_touching):
    ''' Obtain transformation from segment-to-sensor

    Args:
        + data_static (dict of pd.DataFrame): static data for the vertical axis
        + data_walking (dict of pd.DataFrame): walking data for thigh/shank/foot rotational axis
        + data_squat (dict of pd.DataFrame): squat data for pelvis rotational axis
        + dir (str): direction of attachment, e.g., mid, high, low, or front

    Returns:
        + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
    '''
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vx         = np.mean(static_acc, axis = 0)
        fx         = vx/norm(vx)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            static_toe_touching_acc = 1*data_toe_touching[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
            # vx = np.mean(static_toe_touching_acc, axis = 0)

            vx_1 = np.mean(static_toe_touching_acc, axis = 0)
            fx_1 = vx_1/norm(vx_1)

            vz = np.cross(fx_1, fx)
            fz = vz/norm(vz)

            vy = np.cross(fz, fx)
            fy = vy/norm(vy)

        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vy = np.cross(pc1_ax, fx)
            fy = vy/norm(vy)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        else:
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)


            if pc1_ax[-1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            if side == 'r':
                vy = np.cross(pc1_ax, fx)
            else:
                vy = np.cross(fx, pc1_ax)
            
            fy = vy/norm(vy)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens



def sensor_to_segment_mt_cali4(data_static, data_toe_touching, data_leg_abduction):
    
    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            static_toe_touching_acc = 1*data_toe_touching[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()

            vy_1 = np.mean(static_toe_touching_acc, axis = 0)
            fy_1 = vy_1/norm(vy_1)

            vz = np.cross(fy, fy_1)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)

        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            leg_abduction_gyr = 1*data_leg_abduction[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            pc1_ax      = get_pc1_ax_mt(leg_abduction_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vz = np.cross(pc1_ax, fy)
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)
        
        else:
            leg_abduction_gyr = 1*data_leg_abduction[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            pc1_ax      = get_pc1_ax_mt(leg_abduction_gyr)


            if pc1_ax[-1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            if side == 'r':
                # vz = np.cross(pc1_ax, fy)
                vz = np.cross(fy, pc1_ax)
            else:
                vz = np.cross(fy, pc1_ax)
            
            fz = vz/norm(vz)

            vx = np.cross(fy, fz)
            fx = vx/norm(vx)

        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens
