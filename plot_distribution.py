import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

INTRA_FILE = r"output\train_intra_matching.csv"
INTER_NEG_FILE = r"output\train_neg_inter_matching.csv"
INTER_POS_FILE = r"output\train_pos_inter_matching.csv"


# inter_neg_df = pd.read_csv(INTER_NEG_FILE)

# data = {}
# for idx, item in inter_neg_df.iterrows():
#     score = round(float(item['inter_neg_score']), 3)
#     if score in data:
#         data[score] += 1
#     else:
#         data[score] = 1

# lists = sorted(data.items()) # sorted by key, return a list of tuples
# x, y = zip(*lists) # unpack a list of pairs into two tuples


# inter_pos_df = pd.read_csv(INTER_POS_FILE)

# data = {}
# for idx, item in inter_pos_df.iterrows():
#     score = round(float(item['inter_pos_score']), 3)
#     if score in data:
#         data[score] += 1
#     else:
#         data[score] = 1

# lists = sorted(data.items()) # sorted by key, return a list of tuples
# x_pos, y_pos = zip(*lists) # unpack a list of pairs into two tuples


# intra_df = pd.read_csv(INTRA_FILE)

# intra_data = {}

# for idx, item in intra_df.iterrows():
#     score = round(float(item['intra_score']), 3)
#     if score in intra_data:
#         intra_data[score] += 1
#     else:
#         intra_data[score] = 1


# lists = sorted(intra_data.items()) # sorted by key, return a list of tuples
# x_intra, y_intra = zip(*lists) # unpack a list of pairs into two tuples

# plt.plot(x_pos, y_pos, color='blue', label="Positive-Inter distribution")
# plt.plot(x, y, color='black', label="Negative-Inter distribution")
# plt.plot(x_intra, y_intra, color='red', label="Intra distribution")
# plt.legend(loc="upper left")
# plt.show()



def data_analysis(th=0.):

    OUTPUT_FILE = r'output\sorted_train_matching.csv'
    df = pd.read_csv(OUTPUT_FILE)

    data = []
    for idx, item in df.iterrows():
        score = round(float(item['intra_score']) - max(float(item['inter_neg_score']), float(item['inter_pos_score'])), 5)
        if score > th:
            data.append(item)
        

    df_out = pd.DataFrame(data)

    df_out['period'] = df_out.classIDx.astype(str).str.cat(df_out.video_name.astype(str))
    output_data = df_out['period'].to_list()
    class_IDx = df_out['classIDx'].to_list()

    print(th, len(df), len(df_out), len(set(output_data)), len(set(class_IDx)))
    return len(df_out), len(set(output_data)), len(set(class_IDx))

if __name__ == '__main__':

    
    x = np.arange(-0.5, -0.1, 0.05)
    x2 = np.arange(-0.1, 0., 0.005)
    x = np.append(x, x2)
    x = np.append(x, [0.])
    print(x)
    y1, y2, y3 = [], [], []
    for i in x:
        num_frames, num_videos, num_classes = data_analysis(i)
        y1.append(num_frames)
        y2.append(num_videos)
        y3.append(num_classes)

    # y1 = [194001, 193536, 192443 , 189958 , 183144 , 169782 ,142759 ,101384 ,59372 ,55446 ,51848 ,48495 ,45398 ,42301 ,39203 ,36215 ,33440 ,30957 ,28627 ,26251 ,23975 ,21973 ,20086 ,18369 ,16599 ,14876 ,13243 ,11584 ,10156 ]
    # y2 = [2661 ,2661 ,2661 ,2656 ,2640 ,2601 ,2493 ,2132 ,1577 ,1512 ,1447 ,1378 ,1311 ,1255 ,1191 ,1125 ,1060 ,996 ,926 ,856 ,788 ,718 ,659 ,609 ,555 ,501 ,442 , 405 ,353 ]
    # y3 = [243, 243, 243, 243, 243, 243, 242, 239, 209, 203, 199, 191, 187, 182, 180,175,169,165,158,149, 143, 138, 134,126,116, 110,95, 92, 82 ]
    

    # Create figure and axis #1
    fig, ax1 = plt.subplots()
    # plot line chart on axis #1
    p1, = ax1.plot(x, y1, 'g-') 
    ax1.set_ylabel('No frames', color='g')
    # ax1.set_ylim(0, 25)
    # ax1.legend(['average_temp'], loc="upper left")
    # ax1.yaxis.label.set_color(p1.get_color())
    # ax1.yaxis.label.set_fontsize(14)
    # ax1.tick_params(axis='y', colors=p1.get_color(), labelsize=14)
    # set up the 2nd axis
    ax2 = ax1.twinx() 
    # plot bar chart on axis #2
    p2, = ax2.plot(x, y2, 'b-')
    ax2.grid(False) # turn off grid #2
    ax2.set_ylabel('No videos', color='b')
    # ax2.set_ylim(0, 90)
    # ax2.legend(['average_percipitation_mm'], loc="upper center")
    # ax2.yaxis.label.set_color(p2.get_color())
    # ax2.yaxis.label.set_fontsize(14)
    # ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    # set up the 3rd axis
    ax3 = ax1.twinx()
    # Offset the right spine of ax3.  The ticks and label have already been placed on the right by twinx above.
    ax3.spines.right.set_position(("axes", 1.15))
    # Plot line chart on axis #3
    p3, = ax3.plot(x, y3, 'r-')
    ax3.grid(False) # turn off grid #3
    ax3.set_ylabel('No references', color='r')
    # ax3.set_ylim(0, 8)
    # ax3.legend(['average_uv_index'], loc="upper right")
    # ax3.yaxis.label.set_color(p3.get_color())
    # ax3.yaxis.label.set_fontsize(14)
    # ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=14)
    plt.show()



    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, 'g-')
    # ax2.plot(x, y2, 'b-')

    # ax1.set_xlabel('Threshold')
    # ax1.set_ylabel('No frames', color='g')
    # ax2.set_ylabel('No videos', color='b')

    # plt.show()



# lists = sorted(data.items())
# x, y = zip(*lists)
# plt.plot(x, y, color='black', label="{Intra-score} - MAX({Inter-Neg}, {Inter-Pos})")
# plt.legend(loc="upper left")
# plt.show()
