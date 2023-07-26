import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# matching_file = r'output_setD\ggl_matching.csv'
matching_file = r'training_data\siamese_matching.csv'
OUTPUT_FILE = r"iciap_plots\siamese_result.npz"

def plot_precision_recall_curve():
    df = pd.read_csv(matching_file)
    # q_image,q_class,ref_image,ref_class,score

    # thresholds = np.arange(start=0.9, stop=0.9, step=0.025)
    # high_threshold = np.arange(start=0.999, stop=1., step=0.00001)
    # high_threshold = np.arange(start=0.9, stop=0.975, step=0.005)

    thresholds = np.arange(start=0.0, stop=1., step=0.001)
    # high_threshold = np.arange(start=0.91, stop=0.99, step=0.003)
    # thresholds = np.append(thresholds, high_threshold)
    # print("Thresolds:", thresholds)
    
    Precision = []
    Recall = []
    F1_score = []
    TPR = []
    FPR = []

    # return
    N = len(df)
    for th in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0
        
        df_rs = df.loc[df['score']>= th].copy()
        M = len(df_rs)

        df_tp = df_rs.loc[df_rs['q_class'] == df_rs['ref_class']].copy()

        TP = len(df_tp)

        # print(np.round(th, 3), TP, M, N)
                
        pre = np.round(TP /  M, 5)
        rec = np.round(TP / N, 5)

        # pos_df = df.loc[df['q_class'] < 243].copy()

        # neg_df = df.loc[df['q_class'] == 243].copy()

        # M = len(pos_df)
        # TP = len(pos_df.loc[(pos_df['q_class']==pos_df['ref_class']) & (pos_df['score'] >=th)].copy())
        # FN = M - TP

        # N = len(neg_df)
        # TN = len(neg_df.loc[(neg_df['q_class']==neg_df['ref_class']) & (neg_df['score'] >=th)].copy())
        # FP = N - TN

        # pre = np.round((TP/(TP+FP)), 5)
        # rec = np.round((TP/(TP+FN)), 5)

        # tpr = np.round(TP / M,5)
        # fpr = np.round(FP / N,5)

        # TPR.append(tpr)
        # FPR.append(fpr)
        # print("True positive Rate:", tpr, "False positive rate:", fpr)

        f1 = np.round((2*pre*rec) / (pre + rec), 5)
        Precision.append(pre)
        Recall.append(rec)
        F1_score.append(f1)
        # print("Threshold:", np.round(th,5), "TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN, "Precision:", pre, "Recall:", rec, "F1:", f1)


        # print("Threshold:", np.round(th,3), "Precision:", pre, "Recall:", rec, "F1:", f1)

    print("Maximum F1:", np.max(F1_score))
    np.savez(
        OUTPUT_FILE,
        threshold =thresholds,
        precision = Precision,
        recall = Recall,
        F1_score = F1_score
    )
    
    # create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)


    ax1.plot(Recall, Precision, color='blue')    
    # add axis labels to plot
    # ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax1.grid()

    ax2.plot(thresholds, F1_score, color='blue')
    ax2.set_ylabel('F1 score')
    # ax2.set_ylim(top=1)
    ax2.set_xlabel('Threshold')
    # ax2.set_xlim(left=0, right=1)
    ax2.grid()

    #display plot
    plt.show()


if __name__ == '__main__':
    print('[INFO] starting ....')
    plot_precision_recall_curve()
   
    