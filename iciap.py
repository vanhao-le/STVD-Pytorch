import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def cnn_plot():
    ZNCC_RESULT = r"output_setD\zncc_result.npz"
    VGG_RESULT = r"output_setD\vgg16_result.npz"
    RES_RESULT = r"output_setD\rs50_result.npz"
    GGL_RESULT = r"output_setD\ggl_result.npz"

    KF_RESULT = r"output\KF_result.npz"

    # zncc_data = np.load(ZNCC_RESULT)
    # zncc_threshold = zncc_data['threshold']
    # zncc_precision = zncc_data['precision']
    # zncc_recall = zncc_data['recall']
    # zncc_f1 = zncc_data['F1_score']


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

    kf_data = np.load(KF_RESULT)
    kf_threshold = kf_data['threshold']
    kf_precision = kf_data['precision']
    kf_recall = kf_data['recall']
    kf_f1 = kf_data['F1_score']

    ggl_data = np.load(GGL_RESULT)
    ggl_threshold = ggl_data['threshold']
    ggl_precision = ggl_data['precision']
    ggl_recall = ggl_data['recall']
    ggl_f1 = ggl_data['F1_score']


    # #create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(kf_recall, kf_precision, color='red', label="ResNet-50 (Keyfram selection)")
    ax1.plot(rs_recall, rs_precision, color='black', label="ResNet-50 (FPS = 1)")
    # ax1.plot(rs_recall, rs_precision, color='black', label="ResNet-50 (224x224x3, 2048-D)")
    # ax1.plot(ggl_recall, ggl_precision, color='cyan', label="GGL-v3 (299x299x3, 2048-D)")
    # ax1.plot(vgg_recall, vgg_precision, color='red', label="VGG-16 (224x224x3, 4096-D)")
    # ax1.plot(zncc_recall, zncc_precision, color='blue', label="ZNCC (80x60x1, 4800-D)")
    # add axis labels to plot
    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax1.set_ylim(top=1)

    ax1.legend(loc="lower left")
    ax1.grid()

    ax2.plot(kf_threshold, kf_f1, color='red', label="ResNet-50 (Keyfram selection)")
    ax2.plot(rs_threshold, rs_f1, color='black', label="ResNet-50 (FPS = 1)")

    # ax2.plot(rs_threshold, rs_f1, color='black', label="ResNet-50 (224x224x3, 2048-D)")
    # ax2.plot(ggl_threshold, ggl_f1, color='cyan', label="GGL-v3 (299x299x3, 2048-D)")
    # ax2.plot(vgg_threshold, vgg_f1, color='red', label="VGG-16 (224x224x3, 4096-D)")
    # ax2.plot(zncc_threshold, zncc_f1, color='blue', label="ZNCC (80x60x1, 4800-D)")
    # ax2.set_title('Precision-Recall Curve')
    ax2.set_ylabel('F1 score')
    ax2.set_ylim(top=1)
    ax2.set_xlabel('Threshold')
    # ax2.set_xlim(left=0, right=1)
    ax2.legend(loc="lower left")
    ax2.grid()

    #display plot
    plt.show()

def criterion_plot():

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    OUTPUT_FILE = r'iciap_data\sorted_train_matching.csv'
    df = pd.read_csv(OUTPUT_FILE)

    data = {}
    for idx, item in df.iterrows():        
        score = round(item['score'], 3)
        if score in data:
            data[score] += 1
        else:
            data[score] = 1        

    

    lists = sorted(data.items())
    x_frame, y_frame = zip(*lists)
    ax1.plot(x_frame, y_frame, color='black', label='$\phi(X)$ distribution')
    ax1.legend(loc="upper left")
    
    
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
    ax2.legend(loc="upper left")



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
    ax3.legend(loc="upper left")

    
    plt.show()
   

def main():
    criterion_plot()

    print()

if __name__ == '__main__':
    main()
    