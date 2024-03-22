# name: visualizer.py
# description: Visualize results
# author: Vu Phan
# date: 2023/06/05


import matplotlib.pyplot as plt 


def figure_config(size, fontsize, en_box):
    """ Configure figure

    Params: 
        size: size of a figure | list or tuple of 2 int
        fontsize: font size | int
        en_box: enable (1)/disable (0) bounds of the box | dict of 4 bool

    Returns:
        fig, ax: figure and axes
    """
    fig, ax = plt.subplots(figsize = size)
    plt.rcParams.update({'font.size': fontsize})
    
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))

    for axis, value in en_box.items():
        ax.spines[axis].set_visible(value)

    return fig, ax


# TODO: Visualize vertical acceleration of the pelvis IMU
def plot_pelvis_acc_imu():
    """ Plot vertical acceleration of the pelvis IMU (mostly to check the period of the N-pose)

    Params:
        tbd
    
    Returns:
        tbd
    """
    pass # tbd

# TODO: Visualize synchronization
def plot_mocap_imu_sync():
    """ Check the sync'ed results of the mocap and IMU data

    Params:
        tbd
    
    Returns:
        tbd
    """
    pass # tbd


def plot_time_series(mocap, imu, size, fontsize, en_box, title):
    """ Visualize joint angles 

    Params:
        mocap: a joint angle from mocap | np.array
        imu: a joint angle from IMU | np.array
        size: size of a figure | list or tuple of 2 int
        fontsize: font size | int
        en_box: enable (1)/disable (0) bounds of the box | dict of 4 bool
    
    Returns:
        No return, but show a plot
    """    
    fig, ax  = figure_config(size, fontsize, en_box)
    ax.plot(mocap, linewidth = 0.9, alpha = 0.7, label = 'Ground-truth')
    ax.plot(imu, linewidth = 0.9, alpha = 0.7, label = 'IMU')
    ax.set_title(title)
    
    ax.legend()
    plt.show()


def plot_ja_segmented(mocap_ja, imu_ja, segment_id):
    """ Visualize joint angles 

    Params:
        tbd
    
    Returns:
        tbd
    """
    pass # tbd





