import numpy as np
import matplotlib.pyplot as plt

FULL_SEPARABLE = r"iciap_plots\ggl_result.npz"
WORST = r"iciap_plots\vgg16_result.npz"
NOT_CONSITENT = r"iciap_plots\rs50_result.npz"



fs_data = np.load(FULL_SEPARABLE)
fs_threshold = fs_data['threshold']
fs_precision = fs_data['precision']
fs_recall = fs_data['recall']
fs_f1 = fs_data['F1_score']

nc_data = np.load(NOT_CONSITENT)
nc_threshold = nc_data['threshold']
nc_precision = nc_data['precision']
nc_recall = nc_data['recall']
nc_f1 = nc_data['F1_score']

worst_data = np.load(WORST)
worst_threshold = worst_data['threshold']
worst_precision = worst_data['precision']
worst_recall = worst_data['recall']
worst_f1 = worst_data['F1_score']


# #create precision recall curve
fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.plot(worst_recall, worst_precision, color='red', label="W")
# ax1.plot(nc_recall, nc_precision, color='cyan', label="NC")
# ax1.plot(fs_recall, fs_precision, color='black', label="FS")

ax1.plot(worst_recall, worst_precision, color='red', label="VGG-Last FC")
ax1.plot(nc_recall, nc_precision, color='cyan', label="ResNet-Last FC")
ax1.plot(fs_recall, fs_precision, color='black', label="Inception-Last FC")

# print(fs_recall)
# print(fs_precision)

ax1.set_ylabel('Precision')
ax1.set_xlabel('Recall')
ax1.set_ylim(top=1.005)

ax1.legend(loc="lower left")
ax1.grid()

# ax2.plot(worst_threshold, worst_f1, color='red', label="W")
# ax2.plot(nc_threshold, nc_f1, color='cyan', label="NC")
# ax2.plot(fs_threshold, fs_f1, color='black', label="FS")

ax2.plot(worst_threshold, worst_f1, color='red', label="VGG-Last FC")
ax2.plot(nc_threshold, nc_f1, color='cyan', label="ResNet-Last FC")
ax2.plot(fs_threshold, fs_f1, color='black', label="Inception-Last FC")

print(np.max(fs_f1), np.max(nc_f1), np.max(worst_f1))

ax2.set_ylabel('F1 score')
ax2.set_ylim(top=1.01)
ax2.set_xlabel('Threshold')
ax2.set_xlim(left=0.0, right=1)
ax2.legend(loc="lower left")
ax2.grid()

#display plot
plt.show()

