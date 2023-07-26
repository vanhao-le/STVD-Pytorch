import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager


def cnn_plot():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=15)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=13)
    
    VGG_RESULT = r"iciap_plots\vgg16_result.npz"
    RES_RESULT = r"iciap_plots\rs50_result.npz"
    GGL_RESULT = r"iciap_plots\ggl_result.npz"
    MAC_RS50 = r"iciap_plots\MAC_rs50_result.npz"
    MAC_VGG = r"iciap_plots\MAC_vgg16_result.npz"
    RMAC_RS50 = r"iciap_plots\RMAC_rs50_result.npz"
    RMAC_VGG = r"iciap_plots\RMAC_vgg16_result.npz"

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
    ax1.set_xlabel('Threshold \n (a)', fontproperties=font_prop_lable)    
 
    # ax1.set_xlim(0.75, 1.)
    # ax1.set_xlim(0., 1.)
    # ax1.set_ylim(0., 1.0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax1.legend(loc="lower left", prop=font_prop_legend)
     
    ax1.grid(which='major', linestyle='--')

    # marker_size= 5
    # markevery = 0.1

    ax2.plot(rs_recall, rs_precision, color = "black", label="ResNet-50-v1 (Last FC)", linestyle='-')
    ax2.plot(ggl_recall, ggl_precision, color='black', label="Inception-v1 (Last FC)", linestyle='-.')
    ax2.plot(vgg_recall, vgg_precision, color='black', label="VGG-16 (Last FC)", linestyle='--')
    
    
    # add axis labels to plot    
    ax2.set_ylabel('Precision', fontproperties=font_prop_lable)
    ax2.set_xlabel('Recall \n (b)', fontproperties=font_prop_lable)    
 
    # ax2.set_xlim(0.75, 1.)
    # ax2.set_xlim(0., 1.)
    ax2.set_ylim(0.95, 1.0)

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax2.legend(loc="lower left", prop=font_prop_legend)
     
    ax2.grid(which='major', linestyle='--')

    #display plot
    plt.show()

def criterion_plot():
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=12)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.tight_layout()

    # OUTPUT_FILE = r'iciap_data\sorted_train_matching.csv'
    # df = pd.read_csv(OUTPUT_FILE)

    # data = {}
    # for idx, item in df.iterrows():        
    #     score = round(item['score'], 3)
    #     if score in data:
    #         data[score] += 1
    #     else:
    #         data[score] = 1        
    
    # lists = sorted(data.items())
    # x_frame, y_frame = zip(*lists)
    # ax1.plot(x_frame, y_frame, color='black', label='$\phi(X)$ distribution')
    # line_0 = ax1.vlines(x=0., ymin=0, ymax=1000, color='red', linestyle='--', label=r'$\phi(X)=0$')
    
    # # ax1.set_ylim(0., 950)
    # ax1.set_xticks([-0.6, -0.4, -0.2, 0., 0.2])
    # ax1.set_yticks([0, 200, 400, 600, 800, 1000])
    # ax1.set_xlabel('$\phi(X)$ scores \n (a)', fontproperties=font_prop_lable)   

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
        data[i] = count

    data_list = sorted(data.items(), key=lambda x: x[0], reverse=True)
    # print(data_list)   
    x_std, y_std = zip(*data_list)
    ax2.plot(x_std, y_std, color='black', label='$\sigma$ distribution')
    ax2.vlines(x=0.05, ymin=0, ymax=20340, color='gray', linestyle='--')
    ax2.hlines(y=17562, xmin=0, xmax=0.12, color='gray', linestyle='--')
    line_1 = ax2.scatter(0.05, 17562, c='red', marker='s', label= r'$\alpha=0.05$')

    ax2.set_xlabel('$\sigma$ scores \n (b)', fontproperties=font_prop_lable)   
    ax2.set_xticks([0., 0.03, 0.06, 0.09, 0.12])
    ax2.set_yticks([0, 5000, 10000, 15000, 20000])
    # ax2.set_ylim(0, 20380)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    
    ax2.legend(handles=[line_1,], loc="lower right", prop=font_prop_legend) 
    ax2.grid(which='major', linestyle='--')

    # max_list = df['max_score'].to_numpy()

    # error_list = np.arange(-0.7, 0.21, 0.01)
    # # print(error_list)

    # data_max = {}    
    # for i in error_list:
    #     count = 0
    #     for value in max_list:
    #         if value < i:
    #             count += 1
    #     data_max[i] = count
    # max_lst = sorted(data_max.items(), key=lambda x: x[0], reverse=True)
    # # print(np.max(max_lst), np.min(max_lst))
    # # print(data_list)   
    # x_max, y_max = zip(*max_lst)
    # ax3.plot(x_max, y_max, color='black', label='$\mathit{z}_{max}$ distribution')    
   
    # ax3.vlines(x=-0.4, ymin=0, ymax=20340, color='gray', linestyle='--')
    # ax3.hlines(y=100, xmin=-0.6, xmax=0.2, color='gray', linestyle='--')
    # line_2 = ax3.scatter(-0.4, 100, c='red', marker='s', label= r"$\beta=-0.4$")

    # ax3.set_xlabel('$\mathit{z}_{max}$ scores \n (c)', fontproperties=font_prop_lable)   
    # ax3.set_xticks([-0.6, -0.4, -0.2, 0., 0.2])
    # ax3.set_yticks([0, 5000, 10000, 15000, 20000])
    
    # for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    #     label.set_fontproperties(font_prop)
    # ax3.legend(handles=[line_2,], loc="upper left", prop=font_prop_legend)   
    # ax3.grid(which='major', linestyle='--')

    # def on_resize(event):
    #     fig.tight_layout()
    #     fig.canvas.draw()

    # cid = fig.canvas.mpl_connect('resize_event', on_resize)

    plt.show()
    # fig.savefig(r'iciap_data\\kf.png', format='png', dpi=600)

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


def keyframe_plot():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=13)
    
    FS_RESULT = r"iciap_plots\KF_FS_result.npz"
    W_RESULT = r"iciap_plots\KF_Worst_result.npz"
    NC_RESULT = r"iciap_plots\KF_NC_result.npz"
    

    fs_data = np.load(FS_RESULT)
    fs_threshold = fs_data['threshold']
    fs_precision = fs_data['precision']
    fs_recall = fs_data['recall']
    fs_f1 = fs_data['F1_score']

    w_data = np.load(W_RESULT)
    w_threshold = w_data['threshold']
    w_precision = w_data['precision']
    w_recall = w_data['recall']
    w_f1 = w_data['F1_score']


    nc_data = np.load(NC_RESULT)
    nc_threshold = nc_data['threshold']
    nc_precision = nc_data['precision']
    nc_recall = nc_data['recall']
    nc_f1 = nc_data['F1_score']

   

    # #create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)

    
    ax1.plot(fs_recall, fs_precision, color = "black", label="Full separable", linestyle='-')
    ax1.plot(nc_recall, nc_precision, color='black', label="Not consitent", linestyle='-.')
    ax1.plot(w_recall, w_precision, color='black', label="Worst", linestyle='--')
    
    # add axis labels to plot    
    ax1.set_ylabel('Precision', fontproperties=font_prop_lable)
    ax1.set_xlabel('Recall \n (a)', fontproperties=font_prop_lable)    
 
    # ax1.set_xlim(0.75, 1.)
    # ax1.set_xlim(0., 1.)
    # ax1.set_ylim(0., 1.0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax1.legend(loc="lower left", prop=font_prop_legend)
     
    ax1.grid(which='major', linestyle='--')

    marker_size= 5
    markevery = 0.1

    ax2.plot(fs_threshold, fs_f1, color='black', label="Full separable", linestyle='-')
    ax2.plot(nc_threshold, nc_f1, color='black', label="Not consitent", linestyle='-.' )
    ax2.plot(w_threshold, w_f1, color='black', label="Worst", linestyle='--')
    
    # add axis labels to plot    
    ax2.set_ylabel('$F_1$ scores', fontproperties=font_prop_lable)
    ax2.set_xlabel('Thresholds \n (b)', fontproperties=font_prop_lable)    
 
    # ax2.set_xlim(0.75, 1.)
    # ax2.set_xlim(0., 1.)
    # ax2.set_ylim(0., 1.0)

    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontproperties(font_prop)
    ax2.legend(loc="lower left", prop=font_prop_legend)
     
    ax2.grid(which='major', linestyle='--')

    #display plot
    plt.show()

def cnn_plot_comparision():
    
    font_path = r"C:\Windows\Fonts\times.ttf"
    font_prop = font_manager.FontProperties(fname=font_path, size=14)
    font_prop_lable = font_manager.FontProperties(fname=font_path, size=16)
    font_prop_legend = font_manager.FontProperties(fname=font_path, size=13)
    
    VGG_RESULT = r"caip_plots\vgg16_result.npz"
    RES_RESULT = r"caip_plots\rs50_result.npz"
    GGL_RESULT = r"caip_plots\ggl_result.npz"
    MAC_RS50 = r"caip_plots\MAC_rs50_result.npz"
    MAC_VGG = r"caip_plots\MAC_vgg16_result.npz"
    MAC_GGL = r"caip_plots\MAC_ggl_result.npz"
    RMAC_RS50 = r"caip_plots\RMAC_rs50_result.npz"
    RMAC_VGG = r"caip_plots\RMAC_vgg16_result.npz"
    RMAC_GGL = r"caip_plots\RMAC_ggl_result.npz"

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

def main():

    cnn_plot_comparision()
    # cnn_plot()

    # keyframe_plot()


    # criterion_plot()

   

if __name__ == '__main__':
    main()
    