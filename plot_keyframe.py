import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ORIGINAL_FILE = r'output\sorted_train_matching.csv'
FILTERED_FILE = r'output\filtered_train_matching.csv'
PEAKED_1 = r'output\peaked_train_matching_1.csv'
PEAKED_2 = r'output\peaked_train_matching_2.csv'
PEAKED_3 = r'output\peaked_train_matching_3.csv'

# y1, y2, y3 = [], [], []

# df = pd.read_csv(ORIGINAL_FILE)
# df_1 = pd.read_csv(FILTERED_FILE)
# df_2 = pd.read_csv(PEAKED_1)
# df_3 = pd.read_csv(PEAKED_2)
# df_4 = pd.read_csv(PEAKED_3)


# df['period'] = df.classIDx.astype(str).str.cat(df.video_name.astype(str))
# output_data = df['period'].to_list()
# class_IDx = df['classIDx'].to_list()
# y1.append(len(df))
# y2.append(len(set(output_data)))
# y3.append(len(set(class_IDx)))

# df_1['period'] = df_1.classIDx.astype(str).str.cat(df_1.video_name.astype(str))
# output_data = df_1['period'].to_list()
# class_IDx = df_1['classIDx'].to_list()
# y1.append(len(df_1))
# y2.append(len(set(output_data)))
# y3.append(len(set(class_IDx)))



# df_2['period'] = df_2.classIDx.astype(str).str.cat(df_2.video_name.astype(str))
# output_data = df_2['period'].to_list()
# class_IDx = df_2['classIDx'].to_list()
# y1.append(len(df_2))
# y2.append(len(set(output_data)))
# y3.append(len(set(class_IDx)))


# df_3['period'] = df_3.classIDx.astype(str).str.cat(df_3.video_name.astype(str))
# output_data = df_3['period'].to_list()
# class_IDx = df_3['classIDx'].to_list()
# y1.append(len(df_3))
# y2.append(len(set(output_data)))
# y3.append(len(set(class_IDx)))


# df_4['period'] = df_4.classIDx.astype(str).str.cat(df_4.video_name.astype(str))
# output_data = df_4['period'].to_list()
# class_IDx = df_4['classIDx'].to_list()
# y1.append(len(df_4))
# y2.append(len(set(output_data)))
# y3.append(len(set(class_IDx)))


# print(y1, y2, y3)

# # Create figure and axis #1
# fig, ax1 = plt.subplots()
# # plot line chart on axis #1
# p1, = ax1.plot(y1, c="black")
# label_1 = "Original (F: {:d}, V: {:d}, C: {:d})".format(y1[0], y2[0], y3[0])
# label_2 = "Cretirion applied (F: {:d}, V: {:d}, C: {:d})".format(y1[1], y2[1], y3[1])
# label_3 = "Keyframe selection k=1 (F: {:d}, V: {:d}, C: {:d})".format(y1[2], y2[2], y3[2])
# label_4 = "Keyframe selection k=2 (F: {:d}, V: {:d}, C: {:d})".format(y1[3], y2[3], y3[3])
# label_5 = "Keyframe selection k=3 (F: {:d}, V: {:d}, C: {:d})".format(y1[4], y2[4], y3[4])
# # print(lable1)
# ax1.scatter(0, y1[0], c='r', marker="o", label=label_1)
# ax1.scatter(1, y1[1], c='r', marker="+", label=label_2)
# ax1.scatter(2, y1[2], c='r', marker="v", label=label_3)
# ax1.scatter(3, y1[3], c='r', marker="s", label=label_4)
# ax1.scatter(4, y1[4], c='r', marker="x", label=label_5)

# ax1.set_ylabel('No frames', color='black')
# ax1.set_xlabel('Step', color='black')
# ax1.set_xticks([0, 1, 2, 3, 4])
# plt.grid('major')
# ax1.legend(loc="upper right")
# plt.show()





KF_FILE = r'output\keyframe_train_selection.csv'

df = pd.read_csv(KF_FILE)
data = {}
count = 0
for idx, item in df.iterrows():
    classIDx = item['classIDx']
    str_frame_lst = str(item['frame_idx'])
    frame_lst = np.fromstring(str_frame_lst[1:-1], dtype=np.int, sep=',')
    # print(classIDx, "-", frame_lst)
    frame_len = len(frame_lst)
    data[classIDx] = frame_len

    # count += 1
    # if count > 2:
    #     break

data_list = sorted(data.items())

print(data_list)
x, y = zip(*data_list)

plt.plot(x, y)
plt.xlabel("Number of references")
plt.ylabel("Number of frames")
plt.show()




'''
not used

'''


# # ax1.set_ylim(0, 25)
# # ax1.legend(['average_temp'], loc="upper left")
# # ax1.yaxis.label.set_color(p1.get_color())
# # ax1.yaxis.label.set_fontsize(14)
# # ax1.tick_params(axis='y', colors=p1.get_color(), labelsize=14)
# # set up the 2nd axis
# ax2 = ax1.twinx() 
# # plot bar chart on axis #2
# p2, = ax2.plot(y2, 'b-')
# ax2.grid(False) # turn off grid #2
# ax2.set_ylabel('No videos', color='b')
# # ax2.set_ylim(0, 90)
# # ax2.legend(['average_percipitation_mm'], loc="upper center")
# # ax2.yaxis.label.set_color(p2.get_color())
# # ax2.yaxis.label.set_fontsize(14)
# # ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
# # set up the 3rd axis
# ax3 = ax1.twinx()
# # Offset the right spine of ax3.  The ticks and label have already been placed on the right by twinx above.
# ax3.spines.right.set_position(("axes", 1.15))
# # Plot line chart on axis #3
# p3, = ax3.plot(y3, 'r+')
# ax3.grid(False) # turn off grid #3
# ax3.set_ylabel('No references', color='r')
# ax3.set_ylim(0, 8)
# ax3.legend(['average_uv_index'], loc="upper right")
# ax3.yaxis.label.set_color(p3.get_color())
# ax3.yaxis.label.set_fontsize(14)
# ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=14)
# plt.show()