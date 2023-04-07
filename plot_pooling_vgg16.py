import numpy as np
import matplotlib.pyplot as plt


VGG_RESULT = r"output\vgg16_result.npz"
MAC_RESULT = r"output\MAC_vgg16_result.npz"
RMAC_RESULT = r"output\RMAC_vgg16_result.npz"


vgg_data = np.load(VGG_RESULT)
vgg_threshold = vgg_data['threshold']
vgg_precision = vgg_data['precision']
vgg_recall = vgg_data['recall']
vgg_f1 = vgg_data['F1_score']

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
ax1.plot(vgg_recall, vgg_precision, color='black', label="VGG-16 (Last FC) 4096-D")
ax1.plot(mac_recall, mac_precision, color='cyan', label="VGG-16 (MAC) 512-D")
ax1.plot(rmac_recall, rmac_precision, color='red', label="VGG-16 (R-MAC) 512-D")
# ax1.plot(gem_recall, gem_precision, color='blue', label="ResNet-50 (GeM)")
# add axis labels to plot
ax1.set_title('Precision-Recall Curve')
ax1.set_ylabel('Precision')
ax1.set_xlabel('Recall')
ax1.set_ylim(top=1)

ax1.legend(loc="lower left")
ax1.grid()

ax2.plot(vgg_threshold, vgg_f1, color='black', label="VGG-16 (Last FC) 4096-D")
ax2.plot(mac_threshold, mac_f1, color='cyan', label="VGG-16 (MAC) 512-D")
ax2.plot(rmac_threshold, rmac_f1, color='red', label="VGG-16 (R-MAC) 512-D")

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

