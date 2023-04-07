import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import time

FRAME_FILE = r'output\sorted_train_matching.csv'
FILTERED_FILE = r'output\filtered_train_matching.csv'
PEAKED_FILE = r'output\peaked_train_matching_3.csv'
KF_FILE = r'output\keyframe_train_selection.csv'
num_neighbor = 3

def filter_frames(thershold = -0.4):
    df = pd.read_csv(FRAME_FILE)

    data = []
    for idx, item in df.iterrows():
        score = round(float(item['intra_score']) - max(float(item['inter_neg_score']), float(item['inter_pos_score'])), 5)
        if score > thershold:
            data.append(item)        

    df_out = pd.DataFrame(data)

    df_out.to_csv(FILTERED_FILE, index=False, header=True)

    df_out['period'] = df_out.classIDx.astype(str).str.cat(df_out.video_name.astype(str))
    output_data = df_out['period'].to_list()
    class_IDx = df_out['classIDx'].to_list()
    print(len(df), len(df_out), len(set(output_data)), len(set(class_IDx)))    

def peaks_detection(classIDx = 0, video_name = "", num_neighbor=1):
    df = pd.read_csv(FILTERED_FILE)
   
    df_out = df.loc[(df['classIDx'] == classIDx) & (df['video_name'] == video_name)]
    
    # df_out['score'] = round(df_out['intra_score'].astype(float) - max(df_out['inter_neg_score'].astype(float), df_out['inter_pos_score'].astype(float)), 5)
    
    data = []
    for idx, item in df_out.iterrows():       
        item['score'] = round(float(item['intra_score']) - max(float(item['inter_neg_score']), float(item['inter_pos_score'])), 5)
        data.append(item)

    df_out = pd.DataFrame(data)
    
    df_out.sort_values(['frame_idx'], ascending=[True], inplace=True)
    
    # df_out['max'] = df_out.iloc[argrelextrema(df_out.score.values, np.greater_equal, order=1)[0]]['score']

    # plt.scatter(df_out.index, df_out['max'], c='r', marker="x")
    # plt.plot(df_out.index, df_out['score'])
    # plt.show()


    # You can now use the order argument to decide to how many neighbors this comparison must hold 
    rslt_df = df_out.iloc[argrelextrema(df_out.score.values, np.greater_equal, order=num_neighbor)]
            
    # print(rslt_df)
    # print(len(rslt_df))
    return rslt_df

def keyframe_selection():
    df = pd.read_csv(FILTERED_FILE)
    df['period'] = df.classIDx.astype(str) + '-' + df.video_name.astype(str)
    output_data = df['period'].to_list()

    output_data_lst = set(output_data)
    # print(len(output_data_lst))
    count = 0
    appended_data = []
    
    for item in output_data_lst:        
        classIDx = int(str(item).split('-')[0])
        video_name = str(item).split('-')[1]
        rs_df = peaks_detection(classIDx, video_name, num_neighbor)
        # print(item, len(rs_df))
        appended_data.append(rs_df)
        # count += 1
        # if count > 2:
        #     break
    
    final_data = pd.concat(appended_data)
    final_data.to_csv(PEAKED_FILE, index=False, header=True)
    print(final_data)

def keyframe_analysis():

    df = pd.read_csv(PEAKED_FILE)

    N = len(set(df['classIDx']))

    data = []
    
    for i in range(N):
        c_data = df.loc[df['classIDx'] == i]['frame_idx']
        # frame_idx = len(set(c_data))
        # print(frame_idx)
        frame_idx = sorted(list(set(c_data)))
        # frame_idx = np.array(frame_idx, dtype=object)         
        case = {
            'classIDx': i,
            'frame_idx': frame_idx
        }
        data.append(case)

    df_out = pd.DataFrame(data)
    # df_out['frame_idx'] = df_out['frame_idx'].astype('object')
    df_out.to_csv(KF_FILE, index=False, header=True)


def main():
    print("[INFO] starting .........")
    since = time.time()

    # Threshold: -0.4
    # remaning:  192443 2661 243
    # filter_frames(thershold=-0.4)

    # peaks_detection(140, 'c05_20210217104445')

    # keyframe_selection()

    keyframe_analysis()
    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))

if __name__ == '__main__':
    main()
    