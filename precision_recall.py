import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

matching_file = r'output_setD\gglv1_matching.csv'
# matching_file = r'iciap_data\KF_FS_matching.csv'
OUTPUT_FILE = r"caip_plots\MAC_ggl_result.npz"

def plot_precision_recall_curve():
    df = pd.read_csv(matching_file)

    # print(len(df))
    # q_image,q_class,ref_image,ref_class,score

    # thresholds = np.arange(start=0.9, stop=0.9, step=0.025)
    # high_threshold = np.arange(start=0.999, stop=1., step=0.00001)
    # high_threshold = np.arange(start=0.9, stop=0.975, step=0.005)

    thresholds = np.arange(start=0., stop=0.8, step=0.1)
    high_threshold = np.arange(start=0.81, stop=1., step=0.001)
    thresholds = np.append(thresholds, high_threshold)
    # print("Thresolds:", thresholds)
    
    Precision = []
    Recall = []
    F1_score = []

    groundtruth = len(df)

    for th in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        # pos_df = df.loc[df['q_class'] < 243].copy()

        # neg_df = df.loc[df['q_class'] == 243].copy()

        # M = len(pos_df)
        # TP = len(pos_df.loc[(pos_df['q_class']==pos_df['ref_class']) & (pos_df['score'] >=th)].copy())
        # FN = M - TP

        # N = len(neg_df)
        # TN = len(neg_df.loc[(neg_df['q_class']==neg_df['ref_class']) & (neg_df['score'] >=th)].copy())
        # FP = N - TN

        tmp_df = df.loc[df['score'] >= th].copy()

        correctly_ret = len(tmp_df.loc[(tmp_df['q_class'] == tmp_df['ref_class'])].copy())

        retrieved = len(tmp_df)

        pre = np.round((correctly_ret/(retrieved)), 5)

        rec = np.round((correctly_ret / groundtruth), 5)

        # pre = np.round((TP/(TP+FP)), 5)
        # rec = np.round((TP/(TP+FN)), 5)       
        

        # print("True positive Rate:", tpr, "False positive rate:", fpr)

        f1 = np.round((2*pre*rec) / (pre + rec), 5)
        Precision.append(pre)
        Recall.append(rec)
        F1_score.append(f1)
        # print("Threshold:", np.round(th,5), "TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN, "Precision:", pre, "Recall:", rec, "F1:", f1)


        # print("Threshold:", np.round(th,3), "Precision:", pre, "Recall:", rec, "F1:", f1)

    # print("Maximum F1:", np.max(F1_score))

    print("Maximum F1:", np.max(F1_score))
    # np.savez(
    #     OUTPUT_FILE,
    #     threshold =thresholds,
    #     precision = Precision,
    #     recall = Recall,
    #     F1_score = F1_score
    # )
    
    #create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)


    ax1.plot(Recall, Precision, color='blue')    
    # add axis labels to plot
    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax1.set_ylim(bottom=0.95)
    ax1.grid()

    ax2.plot(thresholds, F1_score, color='blue')
    ax2.set_ylabel('F1 score')
    ax2.set_ylim(top=1, )
    ax2.set_xlabel('Threshold')
    # ax2.set_xlim(left=0, right=1)
    ax2.grid()

    #display plot
    plt.show()


if __name__ == '__main__':
    print('[INFO] starting ....')
    plot_precision_recall_curve()
   
    