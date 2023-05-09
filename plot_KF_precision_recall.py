import numpy as np
import matplotlib.pyplot as plt

FULL_SEPARABLE = r"iciap_data\KF_FS_result.npz"
NOT_SEPARABLE = r"iciap_data\KF_NS_result.npz"
WORST = r"iciap_data\KF_Worst_result.npz"



fs_data = np.load(FULL_SEPARABLE)
fs_threshold = fs_data['threshold']
fs_precision = fs_data['precision']
fs_recall = fs_data['recall']
fs_f1 = fs_data['F1_score']

kf_data = np.load(NOT_SEPARABLE)
kf_threshold = kf_data['threshold']
kf_precision = kf_data['precision']
kf_recall = kf_data['recall']
kf_f1 = kf_data['F1_score']

full_data = np.load(WORST)
full_threshold = full_data['threshold']
full_precision = full_data['precision']
full_recall = full_data['recall']
full_f1 = full_data['F1_score']


# #create precision recall curve
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(full_recall, full_precision, color='red', label="W")
ax1.plot(kf_recall, kf_precision, color='cyan', label="NS")
ax1.plot(fs_recall, fs_precision, color='black', label="FS")

# ax1.plot(vgg_recall, vgg_precision, color='red', label="VGG-16 (224x224x3, 4096-D)")
# ax1.plot(zncc_recall, zncc_precision, color='blue', label="ZNCC (80x60x1, 4800-D)")
# add axis labels to plot
ax1.set_title('Precision-Recall Curve')
ax1.set_ylabel('Precision')
ax1.set_xlabel('Recall')
ax1.set_ylim(top=1.)

ax1.legend(loc="lower left")
ax1.grid()

ax2.plot(full_threshold, full_f1, color='red', label="W")
ax2.plot(kf_threshold, kf_f1, color='cyan', label="NS")
ax2.plot(fs_threshold, fs_f1, color='black', label="FS")


ax2.set_ylabel('F1 score')
ax2.set_ylim(top=1.01)
ax2.set_xlabel('Threshold')
ax2.set_xlim(left=0.7, right=1)
ax2.legend(loc="lower left")
ax2.grid()

#display plot
plt.show()

