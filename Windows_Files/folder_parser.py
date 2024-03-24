import os
import segmentation as sg

directory = r'C:\Users\kiddb\Documents\GitHub\WHT-Project\data\Mocap'
subject = 'subject_001'
folder = directory + '\\' + subject

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    for col_no in range(0,3):
        sg.segment(path, col_no)
    
    