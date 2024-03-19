# name: ik_mocap.py
# description: perform IK to obtain joint kinematics from marker-based motion capture data
# author: Vu Phan
# date: 2024//01/27


import numpy as np
import sys, os
from tqdm import tqdm
from numpy.linalg import norm, inv

import sys, os
sys.path.append('/path/to/acl_work')

from utils import constant_common
from utils.mocap import constant_mocap


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

        if side == constant_mocap.BODY_RIGHT:
            vztemp = np.cross(temp_v2, temp_v1)
        elif side == constant_mocap.BODY_LEFT:
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

        if side == constant_mocap.BODY_RIGHT:
            vztempknee = knee_l - knee_m
        elif side == constant_mocap.BODY_LEFT:
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
            mt2 = np.array([mocap_dt[side + 'MT2 X'][i], mocap_dt[side + 'MT2 Y'][i], mocap_dt[side + 'MT2 Z'][i]])
            mt5 = np.array([mocap_dt[side + 'MT5 X'][i], mocap_dt[side + 'MT5 Y'][i], mocap_dt[side + 'MT5 Z'][i]])
        else:
            mt2 = np.array([mocap_dt[side + '2MT X'][i], mocap_dt[side + '2MT Y'][i], mocap_dt[side + '2MT Z'][i]])
            mt5 = np.array([mocap_dt[side + '5MT X'][i], mocap_dt[side + '5MT Y'][i], mocap_dt[side + '5MT Z'][i]])

        cal          = np.array([mocap_dt[side + 'CAL X'][i], mocap_dt[side + 'CAL Y'][i], mocap_dt[side + 'CAL Z'][i]])
        mml          = np.array([mocap_dt[side + 'MML X'][i], mocap_dt[side + 'MML Y'][i], mocap_dt[side + 'MML Z'][i]])
        lml          = np.array([mocap_dt[side + 'LML X'][i], mocap_dt[side + 'LML Y'][i], mocap_dt[side + 'LML Z'][i]])
        ankle_origin = (mml + lml)/2.0

        temp_vec1	= mt2 - cal
        temp_vec2 	= mt5 - cal

        if side == constant_mocap.BODY_RIGHT:
            vy       = np.cross(temp_vec2, temp_vec1)
            temp_vec = lml - mml
        elif side == constant_mocap.BODY_LEFT:
            vy       = np.cross(temp_vec1, temp_vec2)
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

        vy = rps1 - rps2
        vz = rps2 - lps2
        vx = np.cross(vy, vz)

        cord = get_transformation(vx, vy, vz, rps2)
        lab_to_pelvis_cluster.append(cord)

    return lab_to_pelvis_cluster

# For participants: 2
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

        vy = th1 - th2

        if side == constant_mocap.BODY_RIGHT:
            temp_vec = th2 - th3
        elif side == constant_mocap.BODY_LEFT:
            temp_vec = th3 - th2

        vz = np.cross(temp_vec, vy)
        vx = np.cross(vy, vz)

        cord = get_transformation(vx, vy, vz, th2)
        lab_to_thigh.append(cord)

    return lab_to_thigh


# # For participant: 7, 10
# def get_thigh_coords(mocap_dt, side, num_samples):
#     """ Obtain transformations from lab to thigh cluster

#     Params:
#         mocap_dt: data from mocap | pd.DataFrame
#         side: left ('L') or right ('R') side | str
#         opt: 1 for using TH1, 2, and 3; and 2 for using TH2, 3, and 3 | int

#     Returns:
#         lab_to_thigh: transformations from lab to thigh in the entire trial | list of (4x4) np.array
#     """
#     lab_to_thigh = []

#     m1 = 'TH1'
#     m2 = 'TH2'
#     m3 = 'TH3'
#     m4 = 'TH4'

#     for i in range(num_samples):
#         th1 = np.array([mocap_dt[side + m1 + ' X'][i], mocap_dt[side + m1 + ' Y'][i], mocap_dt[side + m1 + ' Z'][i]])
#         th2 = np.array([mocap_dt[side + m2 + ' X'][i], mocap_dt[side + m2 + ' Y'][i], mocap_dt[side + m2 + ' Z'][i]])
#         th3 = np.array([mocap_dt[side + m3 + ' X'][i], mocap_dt[side + m3 + ' Y'][i], mocap_dt[side + m3 + ' Z'][i]])
#         th4 = np.array([mocap_dt[side + m4 + ' X'][i], mocap_dt[side + m4 + ' Y'][i], mocap_dt[side + m4 + ' Z'][i]])

#         if side == constant_mocap.BODY_RIGHT:
#             vx       = th2 - th3
#             temp_vec = th3 - th4
#         elif side == constant_mocap.BODY_LEFT:
#             vx       = th3 - th2
#             temp_vec = th4 - th2

#         vz = np.cross(temp_vec, vx)
#         vy = np.cross(vz, vx)

#         cord = get_transformation(vx, vy, vz, th2)
#         lab_to_thigh.append(cord)

#     return lab_to_thigh


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

        vy = sh1 - sh2

        if side == constant_mocap.BODY_RIGHT:
            temp_vec = sh2 - sh3
        elif side == constant_mocap.BODY_LEFT:
            temp_vec = sh3 - sh2

        vz = np.cross(temp_vec, vy)
        vx = np.cross(vy, vz)

        cord = get_transformation(vx, vy, vz, sh2)
        lab_to_shank.append(cord)

    return lab_to_shank


def get_foot_coords(mocap_dt, side, num_samples):
    """ Obtain transformation matrices from lab to calcn

    Params:
        mocap_dt: data from mocap | pd.DataFrame
        side: left ('L') or right ('R') side | str

    Returns:
        lab_to_foot: transformations from lab to calcn in the entire trial | list of (4x4) np.array
    """
    lab_to_foot = []
    name_flag    = True

    if 'RMT1 X' in mocap_dt.columns:
        pass
    else:
        name_flag = False

    for i in range(num_samples):
        if name_flag:
            mt2 = np.array([mocap_dt[side + 'MT1 X'][i], mocap_dt[side + 'MT1 Y'][i], mocap_dt[side + 'MT1 Z'][i]])
            mt5 = np.array([mocap_dt[side + 'MT5 X'][i], mocap_dt[side + 'MT5 Y'][i], mocap_dt[side + 'MT5 Z'][i]])
        else:
            mt2 = np.array([mocap_dt[side + '1MT X'][i], mocap_dt[side + '1MT Y'][i], mocap_dt[side + '1MT Z'][i]])
            mt5 = np.array([mocap_dt[side + '5MT X'][i], mocap_dt[side + '5MT Y'][i], mocap_dt[side + '5MT Z'][i]])

        cal          = np.array([mocap_dt[side + 'CAL X'][i], mocap_dt[side + 'CAL Y'][i], mocap_dt[side + 'CAL Z'][i]])

        vx        = mt2 - cal
        temp_vec2 = mt5 - cal

        if side == constant_mocap.BODY_RIGHT:
            vy = np.cross(temp_vec2, vx)
        elif side == constant_mocap.BODY_LEFT:
            vy = np.cross(vx, temp_vec2)

        vz       = np.cross(vx, vy)

        coord = get_transformation(vx, vy, vz, cal)
        lab_to_foot.append(coord)

    return lab_to_foot


def get_orientation_mocap(mocap_dt, cluster_use = True):
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
        mocap_orientation['pelvis_mocap']  = get_pelvis_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['femur_r_mocap'] = get_femur_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['tibia_r_mocap'] = get_tibia_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['calcn_r_mocap'] = get_calcn_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['femur_l_mocap'] = get_femur_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['tibia_l_mocap'] = get_tibia_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['calcn_l_mocap'] = get_calcn_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
    else:
        mocap_orientation['pelvis_mocap']         = get_pelvis_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['pelvis_mocap_cluster'] = get_pelvis_cluster_coords(mocap_dt, num_samples)

        mocap_orientation['femur_r_mocap'] = get_femur_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['thigh_r_mocap'] = get_thigh_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['tibia_r_mocap'] = get_tibia_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['shank_r_mocap'] = get_shank_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['calcn_r_mocap'] = get_calcn_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)
        mocap_orientation['foot_r_mocap']  = get_foot_coords(mocap_dt, constant_mocap.BODY_RIGHT, num_samples)

        mocap_orientation['femur_l_mocap'] = get_femur_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['thigh_l_mocap'] = get_thigh_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['tibia_l_mocap'] = get_tibia_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['shank_l_mocap'] = get_shank_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['calcn_l_mocap'] = get_calcn_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)
        mocap_orientation['foot_l_mocap']  = get_foot_coords(mocap_dt, constant_mocap.BODY_LEFT, num_samples)

    return mocap_orientation


# TODO: Get transformations and calibration
def calibration_mocap(mocap_static_orientation, cluster_use = True):
    """ Align frames of body segments w.r.t. the mocap frame (so all joint angles during static trial are 0's)

    Params:
        mocap_static_orientation: orientation of mocap data during the static trial | dict of (4x4) np.array

    Returns:
        cal_orientation: calibration to pelvis | dict of list of (4x4) np.array
    """
    cal_orientation = {'pcluster_to_pelvis': None,
                       'thigh_to_femur_l': None, 'thigh_to_femur_r': None,
                       'shank_to_tibia_l': None, 'shank_to_tibia_r': None,
                       'foot_to_calcn_l': None, 'foot_to_calcn_r': None}

    if cluster_use == False:
        cal_orientation['pcluster_to_pelvis']  = np.identity(4)
        cal_orientation['thigh_to_femur_l']    = np.identity(4)
        cal_orientation['thigh_to_femur_r']    = np.identity(4)
        cal_orientation['shank_to_tibia_l']    = np.identity(4)
        cal_orientation['shank_to_tibia_r']    = np.identity(4)
        cal_orientation['foot_to_calcn_l']     = np.identity(4)
        cal_orientation['foot_to_calcn_r']     = np.identity(4)
    else:
        cal_orientation['pcluster_to_pelvis']  = np.matmul(inv(mocap_static_orientation['pelvis_mocap_cluster'][0]), mocap_static_orientation['pelvis_mocap'][0])
        cal_orientation['thigh_to_femur_l']    = np.matmul(inv(mocap_static_orientation['thigh_l_mocap'][0]), mocap_static_orientation['femur_l_mocap'][0])
        cal_orientation['thigh_to_femur_r']    = np.matmul(inv(mocap_static_orientation['thigh_r_mocap'][0]), mocap_static_orientation['femur_r_mocap'][0])
        cal_orientation['shank_to_tibia_l']    = np.matmul(inv(mocap_static_orientation['shank_l_mocap'][0]), mocap_static_orientation['tibia_l_mocap'][0])
        cal_orientation['shank_to_tibia_r']    = np.matmul(inv(mocap_static_orientation['shank_r_mocap'][0]), mocap_static_orientation['tibia_r_mocap'][0])
        cal_orientation['foot_to_calcn_l']     = np.matmul(inv(mocap_static_orientation['foot_l_mocap'][0]), mocap_static_orientation['calcn_l_mocap'][0])
        cal_orientation['foot_to_calcn_r']     = np.matmul(inv(mocap_static_orientation['foot_r_mocap'][0]), mocap_static_orientation['calcn_r_mocap'][0])

    return cal_orientation


from scipy.spatial.transform import Rotation as R
def transformation_to_angle(trans_mat):
    r = R.from_matrix(trans_mat[0:3, 0:3])

    angle 	= r.as_euler('xyz', degrees = True)
    angle_x	= angle[0]
    angle_y	= angle[1]
    angle_z	= angle[2]

    return angle_x, angle_y, angle_z


def get_angle_bw_2_coords_mocap(segment_1_tracking_cal, segment_1_tracking, segment_2_tracking_cal, segment_2_tracking):
    """ Obtain angles between two frames (represented by two transformation matrices)

    Params:
        segment_1_tracking_cal: pelvis calibration of segment_1_tracking | (4x4) np.array
		segment_1_tracking: moving orientation of the proximal segment | list of N (4x4) np.array
		segment_2_tracking_cal: pelvis calibration of segment_2_tracking | (4x4) np.array
		segment_2_tracking: moving orientation of the distal segment | list of N (4x4) np.array

    Returns:
        angle_arr: moving angles between frame 0 and 1 | (Nx3) np.array
    """
    angle_arr   = []
    num_samples = len(segment_1_tracking)

    for i in range(num_samples):
        segment_1_bone = np.matmul(segment_1_tracking[i], segment_1_tracking_cal)
        segment_2_bone = np.matmul(segment_2_tracking[i], segment_2_tracking_cal)
        joint_rot      = np.matmul(inv(segment_1_bone), segment_2_bone)
        angle          = np.array(transformation_to_angle(joint_rot))

        angle_arr.append(angle)

    angle_arr = np.array(angle_arr)

    return angle_arr


# TODO: Get joint angles
def get_all_ja_mocap(cal_orientation, mocap_orientation, cluster_use = True):
    """ Obtain all (ground-truth) joint angles from the mocap

    Params:
        cal_orientation: sensor-to-body calibration obtained from imu_static_orientation | dict of (1x4) np.array
		imu_orientation: orientation of each IMU (in quaternion) | dict of (Nx4) np.array

    Returns:
        mocap_ja: mocap-based joint angles | dict of np.array
    """
    num_samples = len(mocap_orientation)
    mocap_ja    = {}

    if cluster_use == False:
        temp_hip_l   = get_angle_bw_2_coords_mocap(cal_orientation['pcluster_to_pelvis'], mocap_orientation['pelvis_mocap'], cal_orientation['thigh_to_femur_l'], mocap_orientation['femur_l_mocap'])
        temp_knee_l  = get_angle_bw_2_coords_mocap(cal_orientation['thigh_to_femur_l'], mocap_orientation['femur_l_mocap'], cal_orientation['shank_to_tibia_l'], mocap_orientation['tibia_l_mocap'])
        temp_ankle_l = get_angle_bw_2_coords_mocap(cal_orientation['shank_to_tibia_l'], mocap_orientation['tibia_l_mocap'], cal_orientation['foot_to_calcn_l'], mocap_orientation['calcn_l_mocap'])
    else:
        temp_hip_l   = get_angle_bw_2_coords_mocap(cal_orientation['pcluster_to_pelvis'], mocap_orientation['pelvis_mocap_cluster'], cal_orientation['thigh_to_femur_l'], mocap_orientation['thigh_l_mocap'])
        temp_knee_l  = get_angle_bw_2_coords_mocap(cal_orientation['thigh_to_femur_l'], mocap_orientation['thigh_l_mocap'], cal_orientation['shank_to_tibia_l'], mocap_orientation['shank_l_mocap'])
        temp_ankle_l = get_angle_bw_2_coords_mocap(cal_orientation['shank_to_tibia_l'], mocap_orientation['shank_l_mocap'], cal_orientation['foot_to_calcn_l'], mocap_orientation['foot_l_mocap'])

    mocap_ja['hip_adduction_l']  = constant_common.JA_SIGN['hip_adduction_l']*temp_hip_l[:, 0]
    mocap_ja['hip_rotation_l']   = constant_common.JA_SIGN['hip_rotation_l']*temp_hip_l[:, 1]
    mocap_ja['hip_flexion_l']    = constant_common.JA_SIGN['hip_flexion_l']*temp_hip_l[:, 2]
    # mocap_ja['knee_adduction_l'] = constant_common.JA_SIGN['knee_adduction_l']*temp_knee_l[:, 0]
    # mocap_ja['knee_rotation_l']  = constant_common.JA_SIGN['knee_rotation_l']*temp_knee_l[:, 1]
    mocap_ja['knee_flexion_l']   = constant_common.JA_SIGN['knee_flexion_l']*temp_knee_l[:, 2]
    mocap_ja['ankle_angle_l']    = constant_common.JA_SIGN['ankle_angle_l']*temp_ankle_l[:, 2]

    if cluster_use == False:
        temp_hip_r   = get_angle_bw_2_coords_mocap(cal_orientation['pcluster_to_pelvis'], mocap_orientation['pelvis_mocap'], cal_orientation['thigh_to_femur_r'], mocap_orientation['femur_r_mocap'])
        temp_knee_r  = get_angle_bw_2_coords_mocap(cal_orientation['thigh_to_femur_r'], mocap_orientation['femur_r_mocap'], cal_orientation['shank_to_tibia_r'], mocap_orientation['tibia_r_mocap'])
        temp_ankle_r = get_angle_bw_2_coords_mocap(cal_orientation['shank_to_tibia_r'], mocap_orientation['tibia_r_mocap'], cal_orientation['foot_to_calcn_r'], mocap_orientation['calcn_r_mocap'])
    else:
        temp_hip_r   = get_angle_bw_2_coords_mocap(cal_orientation['pcluster_to_pelvis'], mocap_orientation['pelvis_mocap_cluster'], cal_orientation['thigh_to_femur_r'], mocap_orientation['thigh_r_mocap'])
        temp_knee_r  = get_angle_bw_2_coords_mocap(cal_orientation['thigh_to_femur_r'], mocap_orientation['thigh_r_mocap'], cal_orientation['shank_to_tibia_r'], mocap_orientation['shank_r_mocap'])
        temp_ankle_r = get_angle_bw_2_coords_mocap(cal_orientation['shank_to_tibia_r'], mocap_orientation['shank_r_mocap'], cal_orientation['foot_to_calcn_r'], mocap_orientation['foot_r_mocap'])

    mocap_ja['hip_adduction_r']  = constant_common.JA_SIGN['hip_adduction_r']*temp_hip_r[:, 0]
    mocap_ja['hip_rotation_r']   = constant_common.JA_SIGN['hip_rotation_r']*temp_hip_r[:, 1]
    mocap_ja['hip_flexion_r']    = constant_common.JA_SIGN['hip_flexion_r']*temp_hip_r[:, 2]
    # mocap_ja['knee_adduction_r'] = constant_common.JA_SIGN['knee_adduction_r']*temp_knee_r[:, 0]
    # mocap_ja['knee_rotation_r']  = constant_common.JA_SIGN['knee_rotation_r']*temp_knee_r[:, 1]
    mocap_ja['knee_flexion_r']   = constant_common.JA_SIGN['knee_flexion_r']*temp_knee_r[:, 2]
    mocap_ja['ankle_angle_r']    = constant_common.JA_SIGN['ankle_angle_r']*temp_ankle_r[:, 2]

    return mocap_ja




