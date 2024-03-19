import matplotlib.pyplot as plt
import pandas as pd
from utils import va, synchronization, common, constant_common
import numpy as np


selected_task = 'sts_jump'
subject = 1
start_ja_id = 759
stop_ja_id = 973
side = 'both'


# Load the data
mocap_file = 's' + str(subject) + '_' + selected_task + '_mocap_results.csv'
imu_file = 's' + str(subject) + '_' + selected_task + '_mt_results.csv'
ja_mocap = pd.read_csv(mocap_file)
ja_mt = pd.read_csv(imu_file)

# Turn to numpy array
ja_mocap = ja_mocap.to_numpy()
ja_mt = ja_mt.to_numpy()

key_list = ['hip_adduction_l', 'hip_rotation_l', 'hip_flexion_l', 'knee_adduction_l', 'knee_rotation_l', 'knee_flexion_l', 'ankle_adduction_l', 'ankle_rotation_l', 'ankle_flexion_l', 'hip_adduction_r', 'hip_rotation_r', 'hip_flexion_r', 'knee_adduction_r', 'knee_rotation_r', 'knee_flexion_r', 'ankle_adduction_r', 'ankle_rotation_r', 'ankle_flexion_r']

main_ja_mocap = {}
main_ja_mt = {}

for index in range(ja_mocap.shape[0]):
    main_ja_mocap[key_list[index]] = ja_mocap[index]
    main_ja_mt[key_list[index]] = ja_mt[index]



temp_ja_storage = []

for jk in main_ja_mt.keys():
    # print(jk)
    temp_rmse = common.get_rmse(main_ja_mocap[jk][start_ja_id:stop_ja_id], main_ja_mt[jk][start_ja_id:stop_ja_id])
    temp_ja_storage.append(temp_rmse)

temp_ja_storage = np.array(temp_ja_storage)
temp_ja_storage = temp_ja_storage.reshape((1, len(temp_ja_storage)))
temp_ja_storage = pd.DataFrame(temp_ja_storage)
#add column names
temp_ja_storage.columns = key_list
temp_ja_storage.to_csv('Milestone_3/s' + str(subject) + '_' + selected_task + '_' + side +'_results.csv')


fig, ax = plt.subplots(nrows = 9, ncols = 2, sharex = True)
#change size to 1920 * 1080
fig.set_size_inches(960/50, 2160/50)


ax[0, 0].plot(main_ja_mocap['hip_adduction_l'][start_ja_id:stop_ja_id])
ax[1, 0].plot(main_ja_mocap['hip_rotation_l'][start_ja_id:stop_ja_id])
ax[2, 0].plot(main_ja_mocap['hip_flexion_l'][start_ja_id:stop_ja_id])
ax[3, 0].plot(main_ja_mocap['knee_adduction_l'][start_ja_id:stop_ja_id])
ax[4, 0].plot(main_ja_mocap['knee_rotation_l'][start_ja_id:stop_ja_id])
ax[5, 0].plot(main_ja_mocap['knee_flexion_l'][start_ja_id:stop_ja_id])
ax[6, 0].plot(main_ja_mocap['ankle_adduction_l'][start_ja_id:stop_ja_id])
ax[7, 0].plot(main_ja_mocap['ankle_rotation_l'][start_ja_id:stop_ja_id])
ax[8, 0].plot(main_ja_mocap['ankle_flexion_l'][start_ja_id:stop_ja_id])


ax[0, 1].plot(main_ja_mocap['hip_adduction_r'][start_ja_id:stop_ja_id])
ax[1, 1].plot(main_ja_mocap['hip_rotation_r'][start_ja_id:stop_ja_id])
ax[2, 1].plot(main_ja_mocap['hip_flexion_r'][start_ja_id:stop_ja_id])
ax[3, 1].plot(main_ja_mocap['knee_adduction_r'][start_ja_id:stop_ja_id])
ax[4, 1].plot(main_ja_mocap['knee_rotation_r'][start_ja_id:stop_ja_id])
ax[5, 1].plot(main_ja_mocap['knee_flexion_r'][start_ja_id:stop_ja_id])
ax[6, 1].plot(main_ja_mocap['ankle_adduction_r'][start_ja_id:stop_ja_id])
ax[7, 1].plot(main_ja_mocap['ankle_rotation_r'][start_ja_id:stop_ja_id])
ax[8, 1].plot(main_ja_mocap['ankle_flexion_r'][start_ja_id:stop_ja_id])


ax[0, 0].plot(main_ja_mt['hip_adduction_l'][start_ja_id:stop_ja_id])
ax[1, 0].plot(main_ja_mt['hip_rotation_l'][start_ja_id:stop_ja_id])
ax[2, 0].plot(main_ja_mt['hip_flexion_l'][start_ja_id:stop_ja_id])
ax[3, 0].plot(main_ja_mt['knee_adduction_l'][start_ja_id:stop_ja_id])
ax[4, 0].plot(main_ja_mt['knee_rotation_l'][start_ja_id:stop_ja_id])
ax[5, 0].plot(main_ja_mt['knee_flexion_l'][start_ja_id:stop_ja_id])
ax[6, 0].plot(main_ja_mt['ankle_adduction_l'][start_ja_id:stop_ja_id])
ax[7, 0].plot(main_ja_mt['ankle_rotation_l'][start_ja_id:stop_ja_id])
ax[8, 0].plot(main_ja_mt['ankle_flexion_l'][start_ja_id:stop_ja_id])



ax[0, 1].plot(main_ja_mt['hip_adduction_r'][start_ja_id:stop_ja_id])
ax[1, 1].plot(main_ja_mt['hip_rotation_r'][start_ja_id:stop_ja_id])
ax[2, 1].plot(main_ja_mt['hip_flexion_r'][start_ja_id:stop_ja_id])
ax[3, 1].plot(main_ja_mt['knee_adduction_r'][start_ja_id:stop_ja_id])
ax[4, 1].plot(main_ja_mt['knee_rotation_r'][start_ja_id:stop_ja_id])
ax[5, 1].plot(main_ja_mt['knee_flexion_r'][start_ja_id:stop_ja_id])
ax[6, 1].plot(main_ja_mt['ankle_adduction_r'][start_ja_id:stop_ja_id])
ax[7, 1].plot(main_ja_mt['ankle_rotation_r'][start_ja_id:stop_ja_id])
ax[8, 1].plot(main_ja_mt['ankle_flexion_r'][start_ja_id:stop_ja_id])

#add labels
ax[0, 0].set_ylabel('hip_adduction_l')
ax[1, 0].set_ylabel('hip_rotation_l')
ax[2, 0].set_ylabel('hip_flexion_l')
ax[3, 0].set_ylabel('knee_adduction_l')
ax[4, 0].set_ylabel('knee_rotation_l')
ax[5, 0].set_ylabel('knee_flexion_l')
ax[6, 0].set_ylabel('ankle_adduction_l')
ax[7, 0].set_ylabel('ankle_rotation_l')
ax[8, 0].set_ylabel('ankle_flexion_l')


ax[0, 1].set_ylabel('hip_adduction_r')
ax[1, 1].set_ylabel('hip_rotation_r')
ax[2, 1].set_ylabel('hip_flexion_r')
ax[3, 1].set_ylabel('knee_adduction_r')
ax[4, 1].set_ylabel('knee_rotation_r')
ax[5, 1].set_ylabel('knee_flexion_r')
ax[6, 1].set_ylabel('ankle_adduction_r')
ax[7, 1].set_ylabel('ankle_rotation_r')
ax[8, 1].set_ylabel('ankle_flexion_r')

#add legend
ax[0, 0].legend(['Mocap', 'IMU'])
ax[1, 0].legend(['Mocap', 'IMU'])
ax[2, 0].legend(['Mocap', 'IMU'])
ax[3, 0].legend(['Mocap', 'IMU'])
ax[4, 0].legend(['Mocap', 'IMU'])
ax[5, 0].legend(['Mocap', 'IMU'])
ax[6, 0].legend(['Mocap', 'IMU'])
ax[7, 0].legend(['Mocap', 'IMU'])
ax[8, 0].legend(['Mocap', 'IMU'])

ax[0, 1].legend(['Mocap', 'IMU'])
ax[1, 1].legend(['Mocap', 'IMU'])
ax[2, 1].legend(['Mocap', 'IMU'])
ax[3, 1].legend(['Mocap', 'IMU'])
ax[4, 1].legend(['Mocap', 'IMU'])
ax[5, 1].legend(['Mocap', 'IMU'])
ax[6, 1].legend(['Mocap', 'IMU'])
ax[7, 1].legend(['Mocap', 'IMU'])
ax[8, 1].legend(['Mocap', 'IMU'])

fig.suptitle('Joint Angles Comparison')
# plt.show()

#save the plot in folder

fig.savefig('Milestone_3/s' + str(subject) + '_' + selected_task + '_' + side + '_results.png')




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

# fig.suptitle(' Joint Angles Comparison')
# plt.show()
