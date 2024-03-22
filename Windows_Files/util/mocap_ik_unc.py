# name: mocap_ik_unc.py
# description: Perform (unconstrained) inverse kinematics for the mocap data (modified rizzolio markerset)
# author: Vu Phan
# date: 2023/06/05


import numpy as np 
import sys, os
from tqdm import tqdm
from numpy.linalg import norm, inv

sys.path.append('/path/to/IMU_Kinematics_Comparison')
from constants.mocap_const import *
from constants.common_const import BODY_LEFT, BODY_RIGHT
from util.rot_matrix_handling import *


# TODO: Anatomical coords
def get_transformation(vx, vy, vz, origin):
    """ Obtain the transformation matrix

    Params:
        vx, vy, vz: axes of the frame | (1x3) np.arrays
        origin: origin of the frame | (1x3) np.array

    Returns:
        transformation: transformation matrix | (4x4) np.array
    """
    fx  = np.append(vx/norm(vx), 0)
    fy  = np.append(vy/norm(vy), 0)
    fz  = np.append(vz/norm(vz), 0)
    pos = np.append(origin, 1)

    transformation = np.transpose([fx, fy, fz, pos])

    return transformation
   

def get_hip_origin(vx, vy, vz, asis, pelvis_depth, pelvis_width, leg_length, side):
    """ Obtain the coordinate of hip origin

    Params:
        vx, vy, vz: axes of the frame | (1x3) np.arrays
        asis: pelvis origin, or midpoint of the RASI and LASI markers | (1x3) np.array
        pelvis_depth: depth of the pelvis | float
        pelvis_length: length of the pelvis | float
        leg_length: length of the leg | float
        side: left ('L') or right ('R') side | str
    
    Returns:
        hip_origin: coordinate of hip origin | (1x3) np.array
    """
    if side == BODY_RIGHT:
        hip_origin = asis + (-0.24*pelvis_depth - 9.9/1000)*vx/norm(vx) + \
                        (-0.16*pelvis_width - 0.04*leg_length - 7.1/1000)*vy/norm(vy) + \
                        (0.28*pelvis_depth + 0.16*pelvis_width + 7.9/1000)*vz/norm(vz)
    elif side == BODY_LEFT:
        hip_origin = asis + (-0.24*pelvis_depth - 9.9/1000)*vx/norm(vx) + \
                        (-0.16*pelvis_width - 0.04*leg_length - 7.1/1000)*vy/norm(vy) - \
                        (0.28*pelvis_depth + 0.16*pelvis_width + 7.9/1000)*vz/norm(vz)
    else:
        pass

    return hip_origin


def get_pelvis_coords(mocap_dt, side, num_samples):
    """Obtain transformation matrices from lab to pelvis
    
    Params:
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
    
    Returns:
        lab_to_pelvis: transformations from lab to pelvis in the entire trial | list of (4x4) np.array
    """
    lab_to_pelvis = []

    for i in range(num_samples):
        rasi = np.array([mocap_dt['RASI X'][i], mocap_dt['RASI Y'][i], mocap_dt['RASI Z'][i]])
        lasi = np.array([mocap_dt['LASI X'][i], mocap_dt['LASI Y'][i], mocap_dt['LASI Z'][i]])
        rpsi = np.array([mocap_dt['RPS2 X'][i], mocap_dt['RPS2 Y'][i], mocap_dt['RPS2 Z'][i]])
        lpsi = np.array([mocap_dt['LPS2 X'][i], mocap_dt['LPS2 Y'][i], mocap_dt['LPS2 Z'][i]])
        psis = (rpsi + lpsi)/2.0
        asis = (rasi + lasi)/2.0
        # gtr  = np.array([mocap_dt[side + 'GTR X'][i], mocap_dt[side + 'GTR Y'][i], mocap_dt[side + 'GTR Z'][i]])
        # lep  = np.array([mocap_dt[side + 'LEP X'][i], mocap_dt[side + 'LEP Y'][i], mocap_dt[side + 'LEP Z'][i]])
        # lml  = np.array([mocap_dt[side + 'LML X'][i], mocap_dt[side + 'LML Y'][i], mocap_dt[side + 'LML Z'][i]])

        vz       = rasi - lasi
        temp_vec = np.cross(vz, rasi - psis)
        vx       = np.cross(temp_vec, vz)
        vy       = np.cross(vz, vx)

        # pelvis_depth = norm(asis - psis)
        # pelvis_width = norm(rasi - lasi)
        # leg_length   = norm(gtr - lep) + norm(lep - lml)
        # hip_origin   = get_hip_origin(vx, vy, vz, asis, pelvis_depth, pelvis_width, leg_length, side)
        hip_origin = 1*asis

        coord = get_transformation(vx, vy, vz, hip_origin)
        lab_to_pelvis.append(coord)

    return lab_to_pelvis


def get_femur_coords(mocap_dt, side, num_samples):
    """ Obtain transformation matrices from lab to femur 

    Params:
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
    
    Returns:
        lab_to_femur: transformations from lab to femur in the entire trial | list of (4x4) np.array
    """
    lab_to_femur = []

    for i in range(num_samples):
        hip    = np.array([mocap_dt[side + 'GTR X'][i], mocap_dt[side + 'GTR Y'][i], mocap_dt[side + 'GTR Z'][i]])
        knee_l = np.array([mocap_dt[side + 'LEP X'][i], mocap_dt[side + 'LEP Y'][i], mocap_dt[side + 'LEP Z'][i]])
        knee_m = np.array([mocap_dt[side + 'MEP X'][i], mocap_dt[side + 'MEP Y'][i], mocap_dt[side + 'MEP Z'][i]])
        knee_o = (knee_l + knee_m)/2.0

        vy      = hip - knee_o
        temp_v1 = hip - knee_l
        temp_v2 = knee_m - knee_l
        vztemp  = np.cross(temp_v1, temp_v2)
        vz      = np.cross(vztemp, vy)
        vx      = np.cross(vy, vz)

        coord = get_transformation(vx, vy, vz, knee_o)
        lab_to_femur.append(coord)
    
    return lab_to_femur


def get_tibia_coords(mocap_dt, side, num_samples):
    """ Obtain transformation matrices from lab to tibia 
    
    Params:
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
    
    Returns:
        lab_to_tibia: transformations from lab to tibia in the entire trial | list of (4x4) np.array
    """
    lab_to_tibia = []

    for i in range(num_samples):
        knee_l  = np.array([mocap_dt[side + 'LEP X'][i], mocap_dt[side + 'LEP Y'][i], mocap_dt[side + 'LEP Z'][i]])
        knee_m  = np.array([mocap_dt[side + 'MEP X'][i], mocap_dt[side + 'MEP Y'][i], mocap_dt[side + 'MEP Z'][i]])
        knee_o  = (knee_l + knee_m)/2.0
        ankle_l = np.array([mocap_dt[side + 'LML X'][i], mocap_dt[side + 'LML Y'][i], mocap_dt[side + 'LML Z'][i]])
        ankle_m = np.array([mocap_dt[side + 'MML X'][i], mocap_dt[side + 'MML Y'][i], mocap_dt[side + 'MML Z'][i]])
        ankle_o = (ankle_l + ankle_m)/2.0

        vy         = knee_o - ankle_o
        vztempknee = knee_m - knee_l 
        vx         = np.cross(vy, vztempknee)
        vz         = np.cross(vx, vy)

        coord = get_transformation(vx, vy, vz, knee_o)
        lab_to_tibia.append(coord)

    return lab_to_tibia


def get_calcn_coords(mocap_dt, side, num_samples):
    """ Obtain transformation matrices from lab to calcn
    
    Params:
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
    
    Returns:
        lab_to_calcn: transformations from lab to calcn in the entire trial | list of (4x4) np.array
    """
    lab_to_calcn = []
    name_flag    = True

    if 'RMT1 X' in mocap_dt.columns:
        pass
    else:
        name_flag = False

    for i in range(num_samples):
        if name_flag:
            mt1 = np.array([mocap_dt[side + 'MT1 X'][i], mocap_dt[side + 'MT1 Y'][i], mocap_dt[side + 'MT1 Z'][i]])
            mt5 = np.array([mocap_dt[side + 'MT5 X'][i], mocap_dt[side + 'MT5 Y'][i], mocap_dt[side + 'MT5 Z'][i]])
        else:
            mt1 = np.array([mocap_dt[side + '1MT X'][i], mocap_dt[side + '1MT Y'][i], mocap_dt[side + '1MT Z'][i]])
            mt5 = np.array([mocap_dt[side + '5MT X'][i], mocap_dt[side + '5MT Y'][i], mocap_dt[side + '5MT Z'][i]])

        cal          = np.array([mocap_dt[side + 'CAL X'][i], mocap_dt[side + 'CAL Y'][i], mocap_dt[side + 'CAL Z'][i]])
        mml          = np.array([mocap_dt[side + 'MML X'][i], mocap_dt[side + 'MML Y'][i], mocap_dt[side + 'MML Z'][i]])
        lml          = np.array([mocap_dt[side + 'LML X'][i], mocap_dt[side + 'LML Y'][i], mocap_dt[side + 'LML Z'][i]])
        ankle_origin = (mml + lml)/2.0

        temp_vec1	= mt1 - cal
        temp_vec2 	= mt5 - cal 
        if side == BODY_RIGHT:
            vy = np.cross(temp_vec2, temp_vec1)
        elif side == BODY_LEFT:
            vy = np.cross(temp_vec1, temp_vec2)
        else: 
            pass 
        temp_vec = mml - lml 
        vx       = np.cross(vy, temp_vec)
        vz       = np.cross(vx, vy)

        coord = get_transformation(vx, vy, vz, ankle_origin)
        lab_to_calcn.append(coord)

    return lab_to_calcn


# TODO: Tracking coords (only for tracking markers)
def get_pelvis_cluster_coords(mocap_dt, num_samples):
    """ Obtain transformation from lab to the pelvis cluster

    Params:
        mocap_dt: data from mocap | pd.DataFrame
        num_samples: number of samples | int
    
    Returns:
        lab_to_pelvis_cluster: transformation from lab to pelvis cluster in the entire trial | list of (4x4) np.array
    """
    lab_to_pelvis_cluster = []

    for i in range(num_samples):
        rps1 = np.array([mocap_dt['RPS1 X'][i], mocap_dt['RPS1 Y'][i], mocap_dt['RPS1 Z'][i]])
        rps2 = np.array([mocap_dt['RPS2 X'][i], mocap_dt['RPS2 Y'][i], mocap_dt['RPS2 Z'][i]])
        lps2 = np.array([mocap_dt['LPS2 X'][i], mocap_dt['LPS2 Y'][i], mocap_dt['LPS2 Z'][i]])

        vx = rps1 - rps2 
        vz = rps2 - lps2 
        vy = np.cross(vz, vx)

        cord = get_transformation(vx, vy, vz, rps2)
        lab_to_pelvis_cluster.append(cord)

    return lab_to_pelvis_cluster

def get_thigh_coords(mocap_dt, side, num_samples, opt = 1):
    """ Obtain transformations from lab to thigh cluster

    Params: 
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
        opt: 1 for using TH1, 2, and 3; and 2 for using TH2, 3, and 3 | int

    Returns:
        lab_to_thigh: transformations from lab to thigh in the entire trial | list of (4x4) np.array
    """
    lab_to_thigh = []

    if opt == 1:
        m1 = 'TH1'
        m2 = 'TH2'
        m3 = 'TH3'
    else:
        m1 = 'TH2'
        m2 = 'TH3'
        m3 = 'TH4'

    for i in range(num_samples):
        th1 = np.array([mocap_dt[side + m1 + ' X'][i], mocap_dt[side + m1 + ' Y'][i], mocap_dt[side + m1 + ' Z'][i]])
        th2 = np.array([mocap_dt[side + m2 + ' X'][i], mocap_dt[side + m2 + ' Y'][i], mocap_dt[side + m2 + ' Z'][i]])
        th3 = np.array([mocap_dt[side + m3 + ' X'][i], mocap_dt[side + m3 + ' Y'][i], mocap_dt[side + m3 + ' Z'][i]])

        vx = th1 - th2 
        vy = th3 - th2 
        vz = np.cross(vx, vy)

        cord = get_transformation(vx, vy, vz, th2)
        lab_to_thigh.append(cord)

    return lab_to_thigh


def get_shank_coords(mocap_dt, side, num_samples, opt = 1):
    """ Obtain transformations from lab to shank cluster

    Params: 
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str
        opt: 1 for using TH1, 2, and 3; and 2 for using TH2, 3, and 3 | int

    Returns:
        lab_to_shank: transformations from lab to shank in the entire trial | list of (4x4) np.array
    """
    lab_to_shank = []
    if opt == 1:
        m1 = 'SH1'
        m2 = 'SH2'
        m3 = 'SH3'
    else:
        m1 = 'SH2'
        m2 = 'SH3'
        m3 = 'SH4'

    for i in range(num_samples):
        sh1 = np.array([mocap_dt[side + m1 + ' X'][i], mocap_dt[side + m1 + ' Y'][i], mocap_dt[side + m1 + ' Z'][i]])
        sh2 = np.array([mocap_dt[side + m2 + ' X'][i], mocap_dt[side + m2 + ' Y'][i], mocap_dt[side + m2 + ' Z'][i]])
        sh3 = np.array([mocap_dt[side + m3 + ' X'][i], mocap_dt[side + m3 + ' Y'][i], mocap_dt[side + m3 + ' Z'][i]])

        vx = sh1 - sh2 
        vy = sh3 - sh2 
        vz = np.cross(vx, vy)

        cord = get_transformation(vx, vy, vz, sh2)
        lab_to_shank.append(cord) 

    return lab_to_shank


def get_orientation_mocap(mocap_dt, cluster_use = False):
    """ Obtain orientation of lower-body segments in the entire trial from mocap data

    Paramas:
        mocap_dt: data from mocap | pd.DataFrame
        cluster_use: anatomical or cluster markers used for IK | bool
    
    Returns:
        mocap_orientation: orientation of body segments | dict of list of (4x4) np.array
    """
    num_samples       = mocap_dt.shape[0]
    mocap_orientation = {'torso_mocap': None, 'pelvis_mocap': None,
						'calcn_r_mocap': None, 'tibia_r_mocap': None, 'femur_r_mocap': None,
						'calcn_l_mocap': None, 'tibia_l_mocap': None, 'femur_l_mocap': None}

    if cluster_use == False:
        mocap_orientation['pelvis_mocap']  = get_pelvis_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['femur_r_mocap'] = get_femur_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['tibia_r_mocap'] = get_tibia_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['calcn_r_mocap'] = get_calcn_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['femur_l_mocap'] = get_femur_coords(mocap_dt, BODY_LEFT, num_samples)
        mocap_orientation['tibia_l_mocap'] = get_tibia_coords(mocap_dt, BODY_LEFT, num_samples)
        mocap_orientation['calcn_l_mocap'] = get_calcn_coords(mocap_dt, BODY_LEFT, num_samples)
    else:
        mocap_orientation['pelvis_mocap']  = get_pelvis_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['pelvis_mocap_cluster'] = get_pelvis_cluster_coords(mocap_dt, num_samples)
        mocap_orientation['femur_r_mocap'] = get_thigh_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['tibia_r_mocap'] = get_shank_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['calcn_r_mocap'] = get_calcn_coords(mocap_dt, BODY_RIGHT, num_samples)
        mocap_orientation['femur_l_mocap'] = get_thigh_coords(mocap_dt, BODY_LEFT, num_samples)
        mocap_orientation['tibia_l_mocap'] = get_shank_coords(mocap_dt, BODY_LEFT, num_samples)
        mocap_orientation['calcn_l_mocap'] = get_calcn_coords(mocap_dt, BODY_LEFT, num_samples)

    return mocap_orientation


# TODO: Get transformations and calibration
def pelvis_calibration_mocap(mocap_static_orientation, cluster_use = False):
    """ Align frames of body segments w.r.t. the mocap frame (so all joint angles during static trial are 0's)

    Params:
        mocap_static_orientation: orientation of mocap data during the static trial | dict of (4x4) np.array
    
    Returns:
        cal_orientation: calibration to pelvis | dict of list of (4x4) np.array
    """
    cal_orientation = {'cal_pelvis': None, 
                        'cal_thigh_l': None, 'cal_thigh_r': None,
                        'cal_shank_l': None, 'cal_shank_r': None, 
                        'cal_foot_l': None, 'cal_foot_r': None} 
    
    if cluster_use == False:
        cal_orientation['cal_pelvis']  = np.matmul(inv(mocap_static_orientation['pelvis_mocap']), mocap_static_orientation['pelvis_mocap'])
    else:
        cal_orientation['cal_pelvis']  = np.matmul(inv(mocap_static_orientation['pelvis_mocap_cluster']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_thigh_l'] = np.matmul(inv(mocap_static_orientation['femur_l_mocap']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_thigh_r'] = np.matmul(inv(mocap_static_orientation['femur_r_mocap']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_shank_l'] = np.matmul(inv(mocap_static_orientation['tibia_l_mocap']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_shank_r'] = np.matmul(inv(mocap_static_orientation['tibia_r_mocap']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_foot_l']  = np.matmul(inv(mocap_static_orientation['calcn_l_mocap']), mocap_static_orientation['pelvis_mocap'])
    cal_orientation['cal_foot_r']  = np.matmul(inv(mocap_static_orientation['calcn_r_mocap']), mocap_static_orientation['pelvis_mocap'])

    return cal_orientation


def get_angle_bw_2_coords_mocap(T0_cal, T0, T1_cal, T1):
    """ Obtain angles between two frames (represented by two transformation matrices)

    Params:
        T0_cal: pelvis calibration of T0 | (4x4) np.array
		T0: moving orientation of frame 0 | list of N (4x4) np.array
		T1_cal: pelvis calibration of T1 | (4x4) np.array
		T1: moving orientation of frame 1 | list of N (4x4) np.array

    Returns:
        angle_arr: moving angles between frame 0 and 1 | (Nx3) np.array
    """
    angle_arr   = []
    num_samples = len(T0)

    for i in range(num_samples):
        T0_star  = np.matmul(T0[i], T0_cal[0])
        T1_star  = np.matmul(T1[i], T1_cal[0])
        # T0_star  = 1*T0[i]
        # T1_star  = 1*T1[i]
        T0_to_T1 = np.matmul(inv(T0_star), T1_star)
        angle    = np.array(transformation_to_angle(T0_to_T1))

        angle_arr.append(angle)

    angle_arr = np.array(angle_arr)

    return angle_arr 


# TODO: Get joint angles
def get_all_ja_mocap(cal_orientation, mocap_orientation, cluster_use = False):
    """ Obtain all (ground-truth) joint angles from the mocap

    Params:
        cal_orientation: sensor-to-body calibration obtained from imu_static_orientation | dict of (1x4) np.array
		imu_orientation: orientation of each IMU (in quaternion) | dict of (Nx4) np.array 

    Returns:
        mocap_ja: mocap-based joint angles | dict of np.array
    """
    num_samples = len(mocap_orientation)
    mocap_ja    = {'hip_rotation_l': None, 'hip_flexion_l': None, 'hip_adduction_l': None, 'knee_angle_l': None, 'ankle_angle_l': None,
			        'hip_rotation_r': None, 'hip_flexion_r': None, 'hip_adduction_r': None, 'knee_angle_r': None, 'ankle_angle_r': None}

    if cluster_use == False:
        temp_hip_l   = get_angle_bw_2_coords_mocap(cal_orientation['cal_pelvis'], mocap_orientation['pelvis_mocap'], cal_orientation['cal_thigh_l'], mocap_orientation['femur_l_mocap'])
    else:
        temp_hip_l   = get_angle_bw_2_coords_mocap(cal_orientation['cal_pelvis'], mocap_orientation['pelvis_mocap_cluster'], cal_orientation['cal_thigh_l'], mocap_orientation['femur_l_mocap'])
    temp_knee_l  = get_angle_bw_2_coords_mocap(cal_orientation['cal_thigh_l'], mocap_orientation['femur_l_mocap'], cal_orientation['cal_shank_l'], mocap_orientation['tibia_l_mocap'])
    temp_ankle_l = get_angle_bw_2_coords_mocap(cal_orientation['cal_shank_l'], mocap_orientation['tibia_l_mocap'], cal_orientation['cal_foot_l'], mocap_orientation['calcn_l_mocap'])
    mocap_ja['hip_rotation_l']  = MOCAP_JA_SIGN['hip_rotation_l']*temp_hip_l[:, 0]
    mocap_ja['hip_flexion_l']   = MOCAP_JA_SIGN['hip_flexion_l']*temp_hip_l[:, 2]
    mocap_ja['hip_adduction_l'] = MOCAP_JA_SIGN['hip_adduction_l']*temp_hip_l[:, 1]
    mocap_ja['knee_angle_l']    = MOCAP_JA_SIGN['knee_angle_l']*temp_knee_l[:, 2]
    mocap_ja['ankle_angle_l']   = MOCAP_JA_SIGN['ankle_angle_l']*temp_ankle_l[:, 2]

    if cluster_use == False:
        temp_hip_r   = get_angle_bw_2_coords_mocap(cal_orientation['cal_pelvis'], mocap_orientation['pelvis_mocap'], cal_orientation['cal_thigh_r'], mocap_orientation['femur_r_mocap'])
    else:
        temp_hip_r   = get_angle_bw_2_coords_mocap(cal_orientation['cal_pelvis'], mocap_orientation['pelvis_mocap_cluster'], cal_orientation['cal_thigh_r'], mocap_orientation['femur_r_mocap'])
    temp_knee_r  = get_angle_bw_2_coords_mocap(cal_orientation['cal_thigh_r'], mocap_orientation['femur_r_mocap'], cal_orientation['cal_shank_r'], mocap_orientation['tibia_r_mocap'])
    temp_ankle_r = get_angle_bw_2_coords_mocap(cal_orientation['cal_shank_r'], mocap_orientation['tibia_r_mocap'], cal_orientation['cal_foot_r'], mocap_orientation['calcn_r_mocap'])
    mocap_ja['hip_rotation_r']  = MOCAP_JA_SIGN['hip_rotation_r']*temp_hip_r[:, 0]
    mocap_ja['hip_flexion_r']   = MOCAP_JA_SIGN['hip_flexion_r']*temp_hip_r[:, 2]
    mocap_ja['hip_adduction_r'] = MOCAP_JA_SIGN['hip_adduction_r']*temp_hip_r[:, 1]	
    mocap_ja['knee_angle_r']    = MOCAP_JA_SIGN['knee_angle_r']*temp_knee_r[:, 2]	
    mocap_ja['ankle_angle_r']   = MOCAP_JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 2]

    return mocap_ja


# TODO: Fuse anatomical and tracking joint angles to have better results
# but first, need to check angles computed by anatomical vs. tracking markers
def fill_gap_mocap(mocap_dt, method, val = 999):
    """ Fill gaps in mocap data to perform low pass filter and downsampling

    Params:
        mocap_dt: mocap data | pd.DataFrame
        method: method chosen for filling gaps | str
        val: if method is iso_constant | int
    
    Returns:
        filled_mocap_dt: mocap data with all gaps filled | pd.DataFrame
    """
    filled_mocap_dt = 1*mocap_dt

    return filled_mocap_dt


