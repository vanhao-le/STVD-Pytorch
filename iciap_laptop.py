import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager
import matplotlib
from  scipy import interpolate
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import rank_filter
from scipy.signal import argrelextrema
import math

def cnn_plot():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=13)
    
    VGG_RESULT = r"iciap_data\vgg16_result.npz"
    RES_RESULT = r"iciap_data\rs50_result.npz"
    GGL_RESULT = r"iciap_data\ggl_result.npz"
    MAC_RS50 = r"iciap_data\MAC_rs50_result.npz"
    MAC_VGG = r"iciap_data\MAC_vgg16_result.npz"
    RMAC_RS50 = r"iciap_data\RMAC_rs50_result.npz"
    RMAC_VGG = r"iciap_data\RMAC_vgg16_result.npz"

    vgg_data = np.load(VGG_RESULT)
    vgg_threshold = vgg_data['threshold']
    vgg_precision = vgg_data['precision']
    vgg_recall = vgg_data['recall']
    vgg_f1 = vgg_data['F1_score']

    rs_data = np.load(RES_RESULT)
    rs_threshold = rs_data['threshold']
    rs_precision = rs_data['precision']
    rs_recall = rs_data['recall']
    rs_f1 = rs_data['F1_score']


    ggl_data = np.load(GGL_RESULT)
    ggl_threshold = ggl_data['threshold']
    ggl_precision = ggl_data['precision']
    ggl_recall = ggl_data['recall']
    ggl_f1 = ggl_data['F1_score']

    mac_rs = np.load(MAC_RS50)
    mac_rs_threshold = mac_rs['threshold']
    mac_rs_f1 = mac_rs['F1_score']

    mac_vgg = np.load(MAC_VGG)
    mac_vgg_threshold = mac_vgg['threshold']
    mac_vgg_f1 = mac_vgg['F1_score']


    rmac_rs = np.load(RMAC_RS50)
    rmac_rs_threshold = rmac_rs['threshold']
    rmac_rs_f1 = rmac_rs['F1_score']

    rmac_vgg = np.load(RMAC_VGG)
    rmac_vgg_threshold = rmac_vgg['threshold']
    rmac_vgg_f1 = rmac_vgg['F1_score']

    # #create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)

    
    ax1.plot(rs_threshold, rs_f1, color = "black", label="ResNet-50-v1 (Last FC)", linestyle='-')
    ax1.plot(ggl_threshold, ggl_f1, color='black', label="Inception-v3 (Last FC)", linestyle='-.')
    ax1.plot(vgg_threshold, vgg_f1, color='black', label="VGG-16 (Last FC)", linestyle='--')
    
    # add axis labels to plot    
    ax1.set_ylabel('$F_1$ scores', fontproperties=font_prop_lable)
    ax1.set_xlabel('Thresholds \n (a)', fontproperties=font_prop_lable)    
 
    ax1.set_xlim(0.75, 1.)
    ax1.set_ylim(0., 1.0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax1.legend(loc="lower left", prop=font_prop_legend)
     
    ax1.grid(which='major', linestyle='--')

    marker_size= 5
    markevery = 0.1

    ax2.plot(rs_threshold, rs_f1, color='black', label="ResNet-50-v1 (Last FC)", linestyle='-', marker='.', ms=marker_size, markevery=markevery)
    ax2.plot(vgg_threshold, vgg_f1, color='black', label="VGG-16 (Last FC)", linestyle='--', marker='.', ms=marker_size, markevery=markevery)

    ax2.plot(mac_rs_threshold, mac_rs_f1, color='black', label="ResNet-50-v1 (MAC)", linestyle='-', marker='>', ms=marker_size, markevery=markevery)
    ax2.plot(mac_vgg_threshold, mac_vgg_f1, color='black', label="VGG-16 (MAC)", linestyle='--', marker='>', ms=marker_size, markevery=markevery)

    ax2.plot(rmac_rs_threshold, rmac_rs_f1, color='black', label="ResNet-50-v1 (R-MAC)", linestyle='-', marker='*', ms=marker_size, markevery=markevery)
    ax2.plot(rmac_vgg_threshold, rmac_vgg_f1, color='black', label="VGG-16 (R-MAC)", linestyle='--', marker='*', ms=marker_size, markevery=markevery)
    
    # add axis labels to plot    
    ax2.set_ylabel('$F_1$ scores', fontproperties=font_prop_lable)
    ax2.set_xlabel('Thresholds \n (b)', fontproperties=font_prop_lable)    
 
    ax2.set_xlim(0.75, 1.)
    ax2.set_ylim(0., 1.0)

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax2.legend(loc="lower left", prop=font_prop_legend)
     
    ax2.grid(which='major', linestyle='--')

    #display plot
    plt.show()

def criterion_plot():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=15)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=17)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=13)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    OUTPUT_FILE = r'iciap_data\sorted_train_matching.csv'
    df = pd.read_csv(OUTPUT_FILE)

    data = {}
    count = 0
    for idx, item in df.iterrows():        
        score = round(item['score'], 3)
        if score > 0:
            count += 1
        if score in data:
            data[score] += 1
        else:
            data[score] = 1
    
    # print("Greater than zero:", count, count / len(df)) # 5 %

    lists = sorted(data.items())
    x_frame, y_frame = zip(*lists)       
   
    ax1.plot(x_frame, y_frame, color='black', label='$\phi(X)$ distribution')
    line_0 = ax1.vlines(x=0., ymin=0, ymax=1000, color='red', linestyle='--', label=r'$\phi(X)=0$')
    
    # ax1.set_ylim(0., 950)
    ax1.set_xticks([-0.6, -0.4, -0.2, 0., 0.2])
    ax1.set_yticks([0, 200, 400, 600, 800, 1000])
    ax1.set_ylabel('Features distribution', fontproperties=font_prop_lable)   
    ax1.set_xlabel('$\phi(X)$ scores \n (a)', fontproperties=font_prop_lable)   

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax1.legend(handles=[line_0 ], loc="upper left", prop=font_prop_legend)
    ax1.grid(which='major', linestyle='--')
    
    
    REFERENCE_FILE = r'iciap_data\reference_characterization.csv'
    df = pd.read_csv(REFERENCE_FILE)
    score_list = df['std_score'].to_numpy()
    x = np.arange(0., 0.13, 0.01)
    data = {}
    for i in x:
        count = 0
        for value in score_list:
            if value < i:
                count += 1
        data[i] = count

    data_list = sorted(data.items(), key=lambda x: x[0], reverse=True)
    # print(data_list)   
    x_std, y_std = zip(*data_list)
    ax2.plot(x_std, y_std, color='black', label='$\sigma$ distribution')
    ax2.vlines(x=0.05, ymin=0, ymax=20340, color='gray', linestyle='--')
    ax2.hlines(y=17562, xmin=0, xmax=0.12, color='gray', linestyle='--')
    line_1 = ax2.scatter(0.05, 17562, c='red', marker='s', label= r'$\alpha=0.05$')

    ax2.set_ylabel('Accumulated index distribution', fontproperties=font_prop_lable)   
    ax2.set_xlabel('$\sigma$ scores \n (b)', fontproperties=font_prop_lable)   
    ax2.set_xticks([0., 0.03, 0.06, 0.09, 0.12])
    ax2.set_yticks([0, 5000, 10000, 15000, 20000])
    # ax2.set_ylim(0, 20380)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    
    ax2.legend(handles=[line_1,], loc="lower right", prop=font_prop_legend) 
    ax2.grid(which='major', linestyle='--')

    max_list = df['max_score'].to_numpy()

    error_list = np.arange(-0.7, 0.21, 0.01)
    # print(error_list)

    data_max = {}    
    for i in error_list:
        count = 0
        for value in max_list:
            if value < i:
                count += 1
        data_max[i] = count
    max_lst = sorted(data_max.items(), key=lambda x: x[0], reverse=True)
    # print(np.max(max_lst), np.min(max_lst))
    # print(data_list)   
    x_max, y_max = zip(*max_lst)
    ax3.plot(x_max, y_max, color='black', label='$\mathit{z}_{max}$ distribution')    
   
    ax3.vlines(x=-0.4, ymin=0, ymax=20340, color='gray', linestyle='--')
    ax3.hlines(y=100, xmin=-0.6, xmax=0.2, color='gray', linestyle='--')
    line_2 = ax3.scatter(-0.4, 100, c='red', marker='s', label= r"$\beta=-0.4$")

    ax3.set_ylabel('Reference distribution', fontproperties=font_prop_lable)   
    ax3.set_xlabel('$\mathit{z}_{max}$ scores \n (c)', fontproperties=font_prop_lable)   
    ax3.set_xticks([-0.6, -0.4, -0.2, 0., 0.2])
    ax3.set_yticks([0, 5000, 10000, 15000, 20000])
    
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax3.legend(handles=[line_2,], loc="upper left", prop=font_prop_legend)   
    ax3.grid(which='major', linestyle='--')

    
    plt.show()
    # fig.savefig(r'iciap_data\\kf.png', format='png', dpi=600)

def criterion_plot_dis():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=15)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=13.7)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=12)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # OUTPUT_FILE = r'iciap_data\sorted_train_matching.csv'
    # df = pd.read_csv(OUTPUT_FILE)

    # data = {}
    # count = 0
    # for idx, item in df.iterrows():        
    #     score = round(item['score'], 3)
    #     if score > 0:
    #         count += 1
    #     if score in data:
    #         data[score] += 1
    #     else:
    #         data[score] = 1
    
    # # print("Greater than zero:", count, count / len(df)) # 5 %

    # lists = sorted(data.items())
    # x_frame, y_frame = zip(*lists)    
    # y_frame = y_frame / np.sum(y_frame)
   
    # ax1.plot(x_frame, y_frame, color='black', label='$\phi(X)$ distribution')
    # line_0 = ax1.vlines(x=0., ymin=0, ymax = 0.005, color='red', linestyle='--', label=r'$\phi(X)=0$')
    
    # # ax1.set_ylim(0., 950)
    # ax1.set_xticks([-0.6, -0.4, -0.2, 0., 0.2])    
    # # ax1.set_yscale('symlog')
    # ax1.set_yticks([0, 0.0025, 0.005])

    # ax1.set_ylabel('Features distribution', fontproperties=font_prop_lable)   
    # ax1.set_xlabel('$\phi(X)$ scores \n (a)', fontproperties=font_prop)   
    
    # for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    #     label.set_fontproperties(font_prop)
    # ax1.legend(handles=[line_0 ], loc="upper left", prop=font_prop_legend)
    # ax1.grid(which='major', linestyle='--')
    
    
    REFERENCE_FILE = r'iciap_data\reference_characterization.csv'
    df = pd.read_csv(REFERENCE_FILE)

    score_list = df['std_score'].to_numpy()
    x = np.arange(0., 0.13, 0.01)
    data = {}
    for i in x:
        count = 0
        for value in score_list:
            if value < i:
                count += 1
        data[i] = count / 20337

    data_list = sorted(data.items(), key=lambda x: x[0], reverse=True)
    # print(data_list)   
    x_std, y_std = zip(*data_list)
    

    ax2.plot(x_std, y_std, color='black', label='$\sigma$ distribution')
    ax2.vlines(x=0.05, ymin=0, ymax=1., color='gray', linestyle='--')
    ax2.hlines(y=0.8635, xmin=0, xmax=0.12, color='gray', linestyle='--')
    line_1 = ax2.scatter(0.05, 0.8635, c='red', marker='s', label= r'$\alpha=0.05$')

    ax2.set_ylabel('Accumulated indices', fontproperties=font_prop_lable)   
    ax2.set_xlabel('$\sigma$ scores \n (b)', fontproperties=font_prop)   
    ax2.set_xticks([0., 0.03, 0.06, 0.09, 0.12])
    ax2.set_yticks([0, 0.5, 1.0])
    # ax2.set_ylim(0, 20380)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    
    ax2.legend(handles=[line_1,], loc="lower right", prop=font_prop_legend) 
    ax2.grid(which='major', linestyle='--')

    

    # error_list = np.arange(-0.6, -0.24, 0.01)
    # # print(error_list)

    # data_max = {}    
    # for i in error_list:
    #     count = 0       
    #     for class_id in range(0, 243):            
    #         df_class = df.loc[df['classIDx'] == class_id]
    #         arr_scores = df_class['min_score'].to_numpy()
    #         max_value = np.max(arr_scores)
    #         if max_value >= i:
    #             count += 1
    #     data_max[i] = count

    # max_lst = sorted(data_max.items(), key=lambda x: x[0], reverse=True)
    # # print(np.max(max_lst), np.min(max_lst))
    # # print(data_list)   
    # x_max, y_max = zip(*max_lst)
    
    # x_arr = []
    # y_arr = []
    # for i in range(0, len(x_max)):
    #     if x_max[i] >= -0.6 and x_max[i] <= -0.24:
    #         x_arr.append(x_max[i])
    #         y_arr.append(y_max[i])
     

    # x_np = np.array(x_arr)
    # y_np = np.array(y_arr)
    # x_np = np.sort(x_np)
    # y_np = np.flip(np.sort(y_np))
    # # print(x_np)
    # # print(y_np)

    

    # new_x_coords = np.linspace(np.min(x_np), np.max(x_np), 50, endpoint=True)
    # s = len(new_x_coords) + math.sqrt(2*len(new_x_coords))
    # print(s)
    # tck = interpolate.splrep(x_np, y_np, k=1, s = 15)       
    # new_y_coords = interpolate.splev(new_x_coords, tck)

    # ax3.plot(new_x_coords, new_y_coords, color='black')    
   
    # ax3.vlines(x=-0.4, ymin=235, ymax=243, color='gray', linestyle='--')
    # ax3.hlines(y=242.4, xmin=-0.6, xmax=-0.23, color='gray', linestyle='--')
    # line_2 = ax3.scatter(-0.4, 242.4, c='red', marker='s', label= r"$\beta=-0.4$")

    # ax3.set_ylabel('Reference distribution', fontproperties=font_prop_lable)   
    # ax3.set_xlabel('$\max(\mathit{z}_{min})$ scores \n (c)', fontproperties=font_prop)   
    # ax3.set_xticks([-0.6, -0.5, -0.4, -0.3])
    # ax3.set_yticks([235, 237, 239, 241, 243])

    # # ax3.set_ylim(235, 243.2)
    # ax3.set_xlim(-0.6, -0.23)
    
    # for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    #     label.set_fontproperties(font_prop)
    # ax3.legend(handles=[line_2,], loc="lower left", prop=font_prop_legend)   
    # ax3.grid(which='major', linestyle='--')

    
    plt.show()
    # fig.savefig(r'iciap_data\\kf.png', format='png', dpi=400)

def high_std_plot():

    INPUT_FILE = r'iciap_data\frame_high_std.csv'
    df = pd.read_csv(INPUT_FILE)

    std_arr = df['std_score'].to_numpy()
    std_min = np.min(std_arr)
    std_max = np.max(std_arr)

    x_values = np.arange(0.05, std_max + 0.01, 0.01)
    
    print(x_values)
    data_pos = {}
    data_neg = {}
    for x_i in x_values:        
        df_pos = df.loc[(df['std_score'] >= x_i) & (df['score'] <= 0) ].copy()
        df_neg = df.loc[(df['std_score'] >= x_i) & (df['score'] > 0) ].copy()
        pos_count = len(df_pos)
        neg_count = len(df_neg)
        data_pos[x_i] = pos_count
        data_neg[x_i] = neg_count
    
    data_list_pos = sorted(data_pos.items(), key=lambda x: x[0])   
    x_1, y_1 = zip(*data_list_pos)

    data_list_neg = sorted(data_neg.items(), key=lambda x: x[0])   
    x_2, y_2 = zip(*data_list_neg)

    plt.bar(x_1, y_1, color ='blue', width = 0.005, label=r'$\phi(X) \leq 0$')
    plt.bar(x_2, y_2, color ='red', width = 0.005, label='$\phi(X) > 0$')
    # plt.vlines(x=0, ymin=0, ymax=14000)
    plt.xlabel("Standard deviation")
    plt.ylabel("Distribution")
    plt.legend(loc="upper right")
    plt.show()


def main():

    # cnn_plot()
    # criterion_plot()
    # criterion_plot_dis()

    high_std_plot()

   

if __name__ == '__main__':
    main()
    