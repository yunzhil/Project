# name: va.python
# description: visual aid for visualization
# author: Vu Phan
# date: 2023/12/12


import numpy as np
import matplotlib.pyplot as plt


# --- Add a frame to a 3D plot --- #
def add_frame_3D(ax, origin, x, y, z, offset = [0, 0, 0], shadow = False):
    ''' add a frame to the 3D plot

    Args:
        + ax: matplotlib.pyplot object for 3D plot
        + origin (np.array): 3D position of the origin
        + x, y, and z (np.array): form a frame from the origin
        + offset (list of float): to translate the frame by [xo, yo, zo]

    Returns:
        + No return
    '''
    dirX = x - origin
    dirY = y - origin
    dirZ = z - origin

    if shadow:
        linestyle = '--'
        alpha     = 0.5
    else:
        linestyle = '-'
        alpha     = 1

    ax.quiver3D(origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2], dirX[0], dirX[1], dirX[2], linewidth = 1.2, color = 'r', linestyle = linestyle, alpha = alpha)
    ax.quiver3D(origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2], dirY[0], dirY[1], dirY[2], linewidth = 1.2, color = 'g', linestyle = linestyle, alpha = alpha)
    ax.quiver3D(origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2], dirZ[0], dirZ[1], dirZ[2], linewidth = 1.2, color = 'b', linestyle = linestyle, alpha = alpha)


# --- Rotate a vector (at the origin) --- #
def rotate_vec(rot_mat, vec):
    ''' Rotate a vector with a rotation matrix

    Args:
        + rot_mat (np.array, shape = 3 x 3): rotation matrix
        + vec (np.array, shape = 3 x 1): vector to be rotated

    Returns:
        + new_vec (np.array, shape = 3 x 1): rotated vector
    '''
    new_vec = np.dot(rot_mat, vec)

    return new_vec


# --- Rotate a frame (at the origin) --- #
def rotate_frame(rot_mat, x, y, z):
    ''' Rotate a frame with a rotation matrix (at the origin)

    Args:
        + rot_mat (np.array, shape = 3 x 3): rotation matrix
        + x, y, and z (np.array, shape = 3 x 1): three vectors defining a frame

    Returns:
        + new_x, new_y, and new_z (np.array, shape = 3 x 1): three rotated vector defining the rotated frame
    '''
    new_x = rotate_vec(rot_mat, x)
    new_y = rotate_vec(rot_mat, y)
    new_z = rotate_vec(rot_mat, z)

    return new_x, new_y, new_z
