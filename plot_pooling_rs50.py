import numpy as np
import matplotlib.pyplot as plt


RES_RESULT = r"output\rs50_result.npz"
MAC_RESULT = r"output\MAC_rs50_result.npz"
RMAC_RESULT = r"output\RMAC_rs50_result.npz"


rs_data = np.load(RES_RESULT)
rs_threshold = rs_data['threshold']
rs_precision = rs_data['precision']
rs_recall = rs_data['recall']
rs_f1 = rs_data['F1_score']

mac_data = np.load(MAC_RESULT)
mac_threshold = mac_data['threshold']
mac_precision = mac_data['precision']
mac_recall = mac_data['recall']
mac_f1 = mac_data['F1_score']

rmac_data = np.load(RMAC_RESULT)
rmac_threshold = rmac_data['threshold']
rmac_precision = rmac_data['precision']
rmac_recall = rmac_data['recall']
rmac_f1 = rmac_data['F1_score']


# #create precision recall curve
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(rs_recall, rs_precision, color='black', label="ResNet-50 (Last FC) 2048-D")
ax1.plot(mac_recall, mac_precision, color='cyan', label="ResNet-50 (MAC) 2048-D")
ax1.plot(rmac_recall, rmac_precision, color='red', label="ResNet-50 (R-MAC) 2048-D")
# ax1.plot(gem_recall, gem_precision, color='blue', label="ResNet-50 (GeM)")
# add axis labels to plot
ax1.set_title('Precision-Recall Curve')
ax1.set_ylabel('Precision')
ax1.set_xlabel('Recall')
ax1.set_ylim(top=1)

ax1.legend(loc="lower left")
ax1.grid()

ax2.plot(rs_threshold, rs_f1, color='black', label="ResNet-50 (Last FC) 2048-D")
ax2.plot(mac_threshold, mac_f1, color='cyan', label="ResNet-50 (MAC) 2048-D")
ax2.plot(rmac_threshold, rmac_f1, color='red', label="ResNet-50 (R-MAC) 2048-D")
# ax2.plot(gem_threshold, gem_f1, color='blue', label="ResNet-50 (GeM)")
# ax2.set_title('Precision-Recall Curve')
ax2.set_ylabel('F1 score')
ax2.set_ylim(top=1)
ax2.set_xlabel('Threshold')
# ax2.set_xlim(left=0, right=1)
ax2.legend(loc="lower left")
ax2.grid()

#display plot
plt.show()

