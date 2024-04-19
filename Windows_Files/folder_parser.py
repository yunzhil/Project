import os
import segment as sg
import ipdb

def group3(data_list):
    group = 0
    marker_list = []
    while group < len(data_list):
        counter = 0
        subgroup = []
        while counter < 3:
            subgroup.append(data_list[group+counter])
            counter += 1
        marker_list.append(subgroup)
        group += 3
    return marker_list


directory = r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\Mocap'
subject = 'subject_001'
folder = directory + '\\' + subject

# parsing order
exercise = ['Hip adduction', 'knee flexion', 'leg swing', 'overground jogging', 'overground walking', 'overground walking toe in', 'over ground walking toe out', 'sts jumping', 'vertical jumping', 'static pose']
seg_side = ['R', 'L']
thigh_marker = ['LEP', 'MEP', 'FBH', 'TH1', 'TH2', 'TH3', 'TH4']
shank_marker = ['TTB', 'LML', 'MML', 'SH1', 'SH2', 'SH3', 'SH4']
foot_marker = ['CAL', 'MT1', 'MT2', 'MT5', 'DP1']
pelvic_marker = ['ASI', 'GTR', 'PS1', 'PS2']
axes = [' X', ' Y', ' Z']
all_markers = []
mode = 'explore' # specific or all

if mode == 'all':
    for side in seg_side:
        for marker in thigh_marker+shank_marker+foot_marker+pelvic_marker:
            for axis in axes:
                all_markers.append(side+marker+axis)
    group3(all_markers)
elif mode == 'specific':
    marker_list = [['RDP1Z', 'LDP1Z'], ['RDP1X', 'LDP1X'], ['RDP1X', 'LDP1X'], ['LCALX', 'RCALX', 'RDP1X', 'LDP1X'], ['RDP1X', 'LDP1X'], ['RDP1X', 'LDP1X'],  ['RASIZ', 'LASIZ', 'RASIY', 'LASIY'], ['RASIY', 'LASIY'], ['APEXY'], ['APEXX', 'APEXY', 'APEXZ']]
elif mode == 'explore':
    marker_group = []
    marker_list = [['DP1'], ['DP1'], ['DP1'], ['CAL', 'DP1'], ['DP1'], ['DP1'], ['ASI'], ['ASI'], ['APEX'], ['APEX']]
    for ind, marker in enumerate(marker_list):
        if 'APEX' not in marker:
            for side in seg_side:
                for axis in axes:
                    marker_group.append(side+marker[0]+axis)
        else:
            for axis in axes:
                marker_group.append(marker[0]+axis)
        marker_list[ind] = marker_group
    for ind, group in enumerate(marker_list):    
        marker_list[ind] = group3(group)

segment_time = [(2300, 3700), (3400, 4800), (2500, 4300), (900, 9100), (1100, 9100), (1000,10000), (900, 9300), (1000, 2900), (1000, 2500), (1000, 2000)]

for index, filename in enumerate(os.listdir(folder)):
    path = os.path.join(folder, filename)
    if mode == 'all':
        for markers in marker_list:
            sg.segment(path, filename, markers, segment_time[index])
    elif mode == 'specific':
        for ind, markers in enumerate(marker_list):
            sg.segment(path, filename, markers, segment_time[index])
    elif mode == 'explore':
        for markers in marker_list[index]:
            sg.segment(path, filename, markers, segment_time[index])

    #elif mode == 'explore':
        #for markers in mar
    