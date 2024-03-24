import os
import segmentation as sg

directory = r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\Mocap'
subject = 'subject_001'
folder = directory + '\\' + subject

# parsing order: Hip adduction, knee flexion, leg swing, overground jogging, overground walking, overground walking toe in, 
# over ground walking toe out, sts jumping, vertical jumping, static pose
segment_time = [(2300, 3700), (3400, 4800), (2500, 4300), (900, 9100), (1100, 9100), (1000,10000), (900, 9300), (1000, 2900), (1000, 2500), (1000, 2000)]

for index, filename in enumerate(os.listdir(folder)):
    path = os.path.join(folder, filename)
    for col_no in range(0,3):
        sg.segment(path, col_no, segment_time[index])
    
    