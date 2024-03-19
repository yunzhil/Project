# name: tuning.py
# description: utils for tuning sensor fusion methods
# author: Vu Phan
# date: 2023/06/06


import numpy as np 


def get_param_mesh(f_type, dim, f_params):
    ''' Get the mesh (or grid) of parameters for searching 

    Args:
        + f_type (str): type of filter, e.g., 'MAD', 'MAH', 'EKF', or 'VQF'
        + dim (str): '9D' or '6D'
        + f_params (list of np.array): parameters of the selected filter, i.e., f_filter
    
    Returns:
        + param_mesh (list of list): each element is a combination of different paramters
    '''
    param_mesh = []
    if f_type == 'MAD':
        param_mesh = 1*f_params['beta']
    elif f_type == 'MAH':
        for p1 in f_params['k_P']:
            for p2 in f_params['k_I']:
                param_mesh.append([p1, p2])
    elif f_type == 'EKF':
        for p1 in f_params['sigma_a']:
            for p2 in f_params['sigma_g']:
                for p3 in f_params['sigma_m']:
                    param_mesh.append([p1, p2, p3])
    elif f_type == 'VQF':
        for p1 in f_params['tauAcc']:
            for p2 in f_params['tauMag']:
                param_mesh.append([p1, p2])

    return param_mesh