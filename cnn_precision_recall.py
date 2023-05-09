import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# matching_file = r'iciap_data\rs50_matching.csv'

matching_file = r'iciap_data\KF_FS_matching.csv'

def plot_precision_recall_curve():
    df = pd.read_csv(matching_file)
    # q_image,q_class,ref_image,ref_class,score

    thresholds = np.arange(start=0., stop=0.9, step=0.1)
    high_threshold = np.arange(start=0.91, stop=1.001, step=0.003)
    thresholds = np.append(thresholds, high_threshold)
    print("Thresolds:", thresholds)
    
    Precision = []
    Recall = []
    F1_score = []

    for th in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0
        for idx, item in df.iterrows():            
            score = float(item['score'])
            q_class = int(item['q_class'])
            ref_class = int(item['ref_class'])
            
            '''
            for q_class < 243
            '''
            if(q_class < 243):
                # q_class = ref_class, if score >= theshold (actual = positive and predicted = positive), 
                # return True Positive, otherwise (score< threshold, actual = positive, predicted = negative) return False Negative
                if (q_class == ref_class):
                    if (score >= th):
                        TP += 1
                    else:
                        FN += 1
                else:
                    # q_class != ref_class, if score >= theshold (actual = negative, predicted = positive)
                    # return False Positive, otherwise (score< threshold, actual = negative, predicted = negative) return True Negative
                    # print(q_class, ref_class, score)
                    if(score >= th):
                        FP += 1
            else:
                # q_class = 243
                if (ref_class < 243 and score >= th):
                    FP += 1

        
        pre = TP / (TP + FP)
        rec = TP / (TP + FN)
        Precision.append(pre)
        Recall.append(rec)
        F1_score.append((2*pre*rec) / (pre + rec))

        print(np.round(th,3), TP, FP, TN, FN, F1_score)


    # #create precision recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2)


    ax1.plot(Recall, Precision, color='blue')    
    # add axis labels to plot
    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    ax1.grid()

    ax2.plot(thresholds, F1_score, color='blue')
    ax2.set_ylabel('F1 score')
    # ax.set_ylim(top=1)
    ax2.set_xlabel('Threshold')
    # ax2.set_xlim(left=0, right=1)
    ax2.grid()

    #display plot
    plt.show()


if __name__ == '__main__':
    print('[INFO] starting ....')
    plot_precision_recall_curve()
   
    