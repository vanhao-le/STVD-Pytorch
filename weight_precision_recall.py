import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# matching_file = r'iciap_data\rs50_matching.csv'

# matching_file = r'iciap_data\KF_FS_matching.csv'
matching_file = r'output_setD\gglv1_matching.csv'
OUTPUT_FILE = r"output_setD\gglv1_result.npz"

# matching_file = r'output_pooling\RMAC_ggl_matching.csv'
# OUTPUT_FILE = r"output_pooling\RMAC_ggl_result.npz"

def plot_precision_recall_curve():
    df = pd.read_csv(matching_file)
    # q_image,q_class,ref_image,ref_class,score

    pos_df = df.loc[df['q_class'] < 243].copy()

    neg_df = df.loc[df['q_class'] == 243].copy() 


    x = np.arange(0., 1.0, 0.001)

    data_intra = {}
    data_inter = {}
    
    for idx, item in pos_df.iterrows():
        score = np.round(item['score'], 4)
        if score in data_intra:
            data_intra[score] += 1
        else:
            data_intra[score] = 1

    for idx, item in neg_df.iterrows():
        score = np.round(item['score'], 4)
        if score in data_inter:
            data_inter[score] += 1
        else:
            data_inter[score] = 1

    # reverse=True
    data_1 = sorted(data_intra.items(), key=lambda x: x[0], )
    data_2 = sorted(data_inter.items(), key=lambda x: x[0], )
    # print(data_list)

    # print(data_list)
    x_1, y_1 = zip(*data_1)
    x_2, y_2 = zip(*data_2)

    plt.plot(x_1, y_1)
    plt.plot(x_2, y_2)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Distribution")
    plt.show()


    # low_thresholds = np.arange(start=0., stop=0.7, step=0.1)
    # # mid_thresholds =  np.arange(start=0.71, stop=0.81, step=0.0)
    # high_threshold = np.arange(start=0.71, stop=0.99, step=0.01)
    # # thresholds = np.append(low_thresholds, mid_thresholds)
    # thresholds = np.append(low_thresholds, high_threshold)
    # print("Thresolds:", thresholds)
    
    # Precision = []
    # Recall = []
    # F1_score = []
    # TP_rate = []
    # FP_rate = []

    # for th in thresholds:
    #     TP, FP, TN, FN = 0, 0, 0, 0      
        
    #     pos_df = df.loc[df['q_class'] < 243].copy()

    #     neg_df = df.loc[df['q_class'] == 243].copy()

    #     M = len(pos_df)
    #     TP = len(pos_df.loc[(pos_df['q_class']==pos_df['ref_class']) & (pos_df['score'] >=th)].copy())
    #     FN = M - TP

    #     N = len(neg_df)
    #     TN = len(neg_df.loc[(neg_df['q_class']==neg_df['ref_class']) & (neg_df['score'] >=th)].copy())
    #     FP = N - TN
        
    #     pre = np.round(TP / (TP + FP), 5)
    #     rec = np.round(TP / (TP + FN), 5)
    #     f1 = np.round((2*pre*rec) / (pre + rec), 5) 
    #     Precision.append(pre)
    #     Recall.append(rec)
    #     F1_score.append(f1)
    #     TP_rate.append(TP / (TP+FN))
    #     FP_rate.append(FP / (FP + TN))
    #     print("Threshold:", np.round(th,5), "TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN, "Precision:", pre, "Recall:", rec, "F1:", f1)


    # print("Maximum F1:", np.max(F1_score))
    # np.savez(
    #     OUTPUT_FILE,
    #     threshold =thresholds,
    #     precision = Precision,
    #     recall = Recall,
    #     F1_score = F1_score
    # )


    # #create precision recall curve
    # fig, (ax1, ax2) = plt.subplots(1, 2)




    # ax1.plot(Recall, Precision, color='blue')    
    # # add axis labels to plot
    # ax1.set_title('Precision-Recall Curve')
    # ax1.set_ylabel('Precision')
    # ax1.set_xlabel('Recall')
    # ax1.grid()

    # ax1.set_ylim(top=1, bottom=0.)

    # ax2.plot(thresholds, F1_score, color='blue')
    # ax2.set_ylabel('F1 score')
    # # ax.set_ylim(top=1)
    # ax2.set_xlabel('Threshold')
    # # ax2.set_xlim(left=0, right=1)
    # ax2.grid()

    #display plot
    # plt.show()


if __name__ == '__main__':
    print('[INFO] starting ....')
    plot_precision_recall_curve()
   
    