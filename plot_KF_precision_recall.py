import numpy as np
import matplotlib.pyplot as plt

RES_RESULT = r"output_setD\rs50_result.npz"
FULL_SEPARABLE = r"keyframe\full_separable_result.npz"
KF_ONE = r"keyframe\KF_one_result.npz"



rs_data = np.load(RES_RESULT)
rs_threshold = rs_data['threshold']
rs_precision = rs_data['precision']
rs_recall = rs_data['recall']
rs_f1 = rs_data['F1_score']

kf_data = np.load(KF_ONE)
kf_threshold = kf_data['threshold']
kf_precision = kf_data['precision']
kf_recall = kf_data['recall']
kf_f1 = kf_data['F1_score']

full_data = np.load(FULL_SEPARABLE)
full_threshold = full_data['threshold']
full_precision = full_data['precision']
full_recall = full_data['recall']
full_f1 = full_data['F1_score']


# #create precision recall curve
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(full_recall, full_precision, color='red', label="ResNet-50 (1 frame / 1 reference)")
ax1.plot(kf_recall, kf_precision, color='cyan', label="ResNet-50 (1 frame / 1 reference)")
ax1.plot(rs_recall, rs_precision, color='black', label="ResNet-50 (FPS = 1)")

# ax1.plot(vgg_recall, vgg_precision, color='red', label="VGG-16 (224x224x3, 4096-D)")
# ax1.plot(zncc_recall, zncc_precision, color='blue', label="ZNCC (80x60x1, 4800-D)")
# add axis labels to plot
ax1.set_title('Precision-Recall Curve')
ax1.set_ylabel('Precision')
ax1.set_xlabel('Recall')
ax1.set_ylim(top=1.)

ax1.legend(loc="lower left")
ax1.grid()

ax2.plot(full_threshold, full_f1, color='red', label="ResNet-50 (Full seperable)")
ax2.plot(kf_threshold, kf_f1, color='cyan', label="ResNet-50 (1 frame / 1 reference)")
ax2.plot(rs_threshold, rs_f1, color='black', label="ResNet-50 (FPS = 1)")


ax2.set_ylabel('F1 score')
ax2.set_ylim(top=1.01)
ax2.set_xlabel('Threshold')
ax2.set_xlim(left=0.7, right=1)
ax2.legend(loc="lower left")
ax2.grid()

#display plot
plt.show()

