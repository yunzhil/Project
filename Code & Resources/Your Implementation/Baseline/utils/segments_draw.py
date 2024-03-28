import numpy as np
import matplotlib.pyplot as plt


s_unc   = []
s_xsens = []
s_os    = []
for i in range(len(segment_id) - 1):
    N = segment_id[i+1] - segment_id[i]
    # s_os.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_os_ja[segment_id[i]:segment_id[i+1]]))
    s_xsens.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), xsens_ja[segment_id[i]:segment_id[i+1]]))
    s_unc.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, N), imu_ja_sel[segment_id[i]:segment_id[i+1]]))
# df_mocap = pd.DataFrame(np.array(s_mocap))
# df_xsens = pd.DataFrame(np.array(s_xsens))
# df_mocap.to_csv('s4_mocap_ankle.csv')
# df_xsens.to_csv('s4_xsens_ankle_unc_misp.csv')
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize = (10,3))
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_unc, axis = 0), linewidth = 1.8, linestyle = '--', color = 'k', label = 'Xsens baseline')
# ax.plot(100*np.linspace(0, 1, 100), np.mean(s_os, axis = 0), linewidth = 1.8, color = '#596F62', label = 'OpenSense biomechanical model')
ax.plot(100*np.linspace(0, 1, 100), np.mean(s_xsens, axis = 0), linewidth = 1.8, color = '#70161E', label = 'Xsens biomechanical model')
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_unc, axis = 0)-2*np.std(s_unc, axis = 0)), (np.mean(s_unc, axis = 0)+2*np.std(s_unc, axis = 0)), color = 'k', alpha=.13)
# ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_os, axis = 0)-2*np.std(s_os, axis = 0)), (np.mean(s_os, axis = 0)+2*np.std(s_os, axis = 0)), color = '#596F62', alpha=.13)
ax.fill_between(100*np.linspace(0, 1, 100), (np.mean(s_xsens, axis = 0)-2*np.std(s_xsens, axis = 0)), (np.mean(s_xsens, axis = 0)+2*np.std(s_xsens, axis = 0)), color = '#70161E', alpha=.13)
ax.set_ylim([0, 120])
# ax.set_ylim([-40, 60])
# ax.set_ylim([-40, 40])
ax.set_xlim([0, 100])
# ax.set_xlabel('Gait cycle')
# ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
# ax.set_ylabel('Angle $(^o)$')
ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc = 'upper left', frameon = False, prop={'size': 18}, ncol = 1)
plt.show()
# plt.savefig("eni_slide_legend.png", bbox_inches='tight')
print('<-- Done')