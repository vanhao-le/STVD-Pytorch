import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib 

# plt.style.use('classic')

# plt.rcParams["font.family"] = "Times New Roman"
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 

KF_FILE = r'output\keyframe_train_selection.csv'
CATE_FILE = r'data\category.csv'


def get_reference_duration():
    df = pd.read_csv(KF_FILE)

    df_ref = pd.read_csv(CATE_FILE)

    data = {}
    count = 0
    for idx, item in df.iterrows():
        classIDx = int(item['classIDx'])
        duration = df_ref.loc[df_ref['classIDx']== classIDx]['duration_sec'].to_list()
        
        str_frame_lst = str(item['frame_idx'])
        frame_lst = np.fromstring(str_frame_lst[1:-1], dtype = np.int32, sep=',')
        # print(classIDx, "-", frame_lst)
        frame_len = len(frame_lst)
        fps = np.round(frame_len / duration[0], 2)
        data[classIDx] = fps

        # count += 1
        # if count > 2:
        #     break

    '''
    The value of the key parameter should be a function (or other callable) that takes a single argument and returns a key to use for sorting purposes. 
    This technique is fast because the key function is called exactly once for each input record.
    In this case, the inline function lambda x: x[1] is defined as a value of the key parameter. 
    The lambda function takes input x return x[1] which is the second element of x.
    '''
    
    data_list = sorted(data.items(), key=lambda x: x[1], reverse=True)
    # print(data_list)

    # print(data_list)
    x, y = zip(*data_list)

    mean = np.mean(y)
    
    plt.hlines(y=mean, xmin=0, xmax=243, colors='red')
    plt.plot(y)
    plt.xlabel("Number of references")
    plt.ylabel("Average FPS")
    plt.show()

def plot_criterion():
    class_id = 8
    FRAME_FILE = r'output\sorted_train_matching.csv'

    df = pd.read_csv(FRAME_FILE)
    df = df.loc[df['classIDx'] == class_id]

    df_videos = set(df['video_name'].to_list())

    # print(df_videos, len(df_videos))

    data = []
    N = len(df_videos)

    for v_name in df_videos:
        df_frame = df.loc[df['video_name'] == v_name]
        score_lst = []
        for idx, item in df_frame.iterrows():
            score = round(float(item['intra_score']) - max(float(item['inter_neg_score']), float(item['inter_pos_score'])), 5)
            score_lst.append(score)
        data.append(score_lst)   

    for i in range(len(data)):
        plt.plot(data[i], label = "video {:d}".format(i))
    
    plt.xlabel("Duration")
    plt.ylabel("Score distribution")
    plt.legend(loc="lower left")
    plt.show()

def plot_mean_std_error():
    FRAME_FILE = r'output\sorted_train_matching.csv'
    df = pd.read_csv(FRAME_FILE)

    class_lst = np.arange(0, 243)
    data = []
    data_b = []
    
    for class_id in class_lst:

        df_class = df.loc[df['classIDx'] == class_id]
        df_videos = set(df_class['video_name'].to_list())

        # print(df_videos, len(df_videos))
        
        '''
        N - numbers of videos
        M - numbers of indexes
        '''
        # N = len(df_videos)
        # M = df_class.loc[df_class['video_name'] == list(df_videos)[0]]['frame_idx'].to_list()

        arr_scores = df_class['score'].to_numpy()
        data.append(np.mean(arr_scores))
        data_b.append(np.std(arr_scores))

    '''
    #use errorbar function to create symmetric horizontal error bar 
    #xerr provides error bar length
    #fmt specifies plot icon for mean value
    #ms= marker size
    #mew = marker thickness
    #capthick = thickness of error bar end caps
    #capsize = size of those caps
    '''

    plt.errorbar(class_lst, data, data_b, color='red', marker='s', linestyle='None', ecolor='black', ms=2, mew=2, capthick=2, capsize=2)
    plt.hlines(y=0, xmin=0, xmax=242, linestyles ="dashed", colors ="gray")
    plt.hlines(y=-0.4, xmin=0, xmax=242, linestyles ="dashed", colors ="red")
    plt.xlabel("References")
    plt.ylabel("Error scores")
    # plt.legend(loc="lower right")
    plt.show()


def plot_error():
    
    FRAME_FILE = r'output\sorted_train_matching.csv'
    OUTPUT_FILE = r'output\mean_std_indexes.csv'
    df = pd.read_csv(FRAME_FILE)

    class_lst = np.arange(0, 243)
    data = []
    data_b = []
    N = 0
    data_c = []
    for class_id in class_lst:

        df_class = df.loc[df['classIDx'] == class_id]
        df_videos = set(df_class['video_name'].to_list())

        # print(df_videos, len(df_videos))
        
        '''
        N - numbers of videos
        M - numbers of indexes
        '''
        index_lst = df_class.loc[df_class['video_name'] == list(df_videos)[0]]['frame_idx'].to_list()
        N += len(index_lst)
        for idx in index_lst:
            arr_scores = df_class.loc[df_class['frame_idx'] == idx]['score'].to_numpy()   
            mean_score = round(np.mean(arr_scores), 5)
            std_score =  round(np.std(arr_scores), 5)
            data.append(mean_score)
            data_b.append(std_score)
            case = {
                'classIDx': class_id,
                'frame_idx': idx,
                'mean_score': mean_score,
                'std_score': std_score

            }
            data_c.append(case)
    

    df_rs = pd.DataFrame(data_c)
    df_rs.to_csv(OUTPUT_FILE, index=False, header=True)

    
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.plot(x, data)
    # ax2.plot(x, data_b)
    
    # ax1.set_xlabel("Indexes")
    # ax1.set_ylabel("Mean Score Error")

    # ax2.set_xlabel("Indexes")
    # ax2.set_ylabel("Std Score Error")
    
    # # plt.legend(loc="lower right")
    # plt.show()

def plot_index_error():

    FRAME_FILE = r'output\sorted_train_matching.csv'
    df = pd.read_csv(FRAME_FILE)

    class_lst = np.arange(52, 53)
    data = []
    data_b = []
    M = []
    
    for class_id in class_lst:

        df_class = df.loc[df['classIDx'] == class_id]
        df_videos = set(df_class['video_name'].to_list())

        # print(df_videos, len(df_videos))
        
        '''
        N - numbers of videos
        M - numbers of indexes
        '''
        # N = len(df_videos)
        M = df_class.loc[df_class['video_name'] == list(df_videos)[0]]['frame_idx'].to_list()

        for i in M:
            df_indexs = df_class.loc[df_class['frame_idx'] == i]
            arr_scores = df_indexs['score'].to_numpy()
            data.append(np.mean(arr_scores))
            data_b.append(np.std(arr_scores))

    '''
    #use errorbar function to create symmetric horizontal error bar 
    #xerr provides error bar length
    #fmt specifies plot icon for mean value
    #ms= marker size
    #mew = marker thickness
    #capthick = thickness of error bar end caps
    #capsize = size of those caps
    '''
    # x = np.arange(M)
    plt.errorbar(M, data, data_b, color='red', marker='s', linestyle='None', ecolor='black', ms=2, mew=2, capthick=2, capsize=2)
    plt.hlines(y=0, xmin=0, xmax=np.max(M), linestyles ="dashed", colors ="gray")
    plt.hlines(y=-0.4, xmin=0, xmax=np.max(M), linestyles ="dashed", colors ="red")
    plt.xlabel("Indexes")
    plt.ylabel("Error scores")
    # plt.legend(loc="lower right")
    plt.show()


def generate_reference_mean_std():
    FRAME_FILE = r'output\sorted_train_matching.csv'
    OUTPUT_FILE = r'output\reference_characterization.csv'
    df = pd.read_csv(FRAME_FILE)

    class_lst = np.arange(0, 243)
    data = []
    
    for class_id in class_lst:

        df_class = df.loc[df['classIDx'] == class_id]
        df_videos = set(df_class['video_name'].to_list())

        # print(df_videos, len(df_videos))
        
        '''
        N - numbers of videos
        M - numbers of indexes
        '''
        # N = len(df_videos)
        M = df_class.loc[df_class['video_name'] == list(df_videos)[0]]['frame_idx'].to_list()

        for i in M:
            df_indexs = df_class.loc[df_class['frame_idx'] == i]
            arr_scores = df_indexs['score'].to_numpy()
            mean_score = round(np.mean(arr_scores), 5)
            std_score = round(np.std(arr_scores),5)
            min_score = round(np.min(arr_scores),5)
            max_score = round(np.max(arr_scores),5)
            case = {
                'classIDx': class_id,
                'frame_idx': i,
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
            }
            data.append(case)
    

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(OUTPUT_FILE, index=False, header=True)

def plot_reference():

    OUTPUT_FILE = r'output\reference_characterization.csv'
    df = pd.read_csv(OUTPUT_FILE)

    N = len(df.index)
    
    mean_array = df['mean_score'].to_numpy()
    std_array = df['std_score'].to_numpy()

    x = np.arange(N)

    plt.errorbar(x, mean_array, std_array, color='red', marker='s', linestyle='None', ecolor='black', ms=1, mew=1, capthick=1, capsize=1)
    plt.hlines(y=0, xmin=0, xmax=N, linestyles ="dashed", colors ="gray")
    plt.hlines(y=-0.4, xmin=0, xmax=N, linestyles ="dashed", colors ="red")
    plt.xlabel("Indexes")
    plt.ylabel("Error scores")
    # plt.legend(loc="lower right")
    plt.show()

def join_sorted_score_reference_characterize():

    FRAME_FILE = r'output\sorted_train_matching.csv'
    REFERENCE_FILE = r'output\reference_characterization.csv'
    OUTPUT_FILE = r'output\reference_joined_score.csv'
    df = pd.read_csv(FRAME_FILE)

    df_ref = pd.read_csv(REFERENCE_FILE)

    for idx, item in df.iterrows():
        a_class = item['classIDx']
        a_frameID = item['frame_idx']
        df_tmp = df_ref.loc[(df_ref['classIDx']== a_class) & (df_ref['frame_idx']== a_frameID)]        
        df.loc[idx, 'mean_score'] = df_tmp['mean_score'].to_numpy()        
        df.loc[idx, 'std_score'] = df_tmp['std_score'].to_numpy()
        df.loc[idx, 'min_score'] = df_tmp['min_score'].to_numpy()
        df.loc[idx, 'max_score'] = df_tmp['max_score'].to_numpy()

    df.to_csv(OUTPUT_FILE, index=False, header=True)


def save_reference_characterization():
    REFERENCE_FILE = r'output\reference_characterization.csv'
    OUTPUT_FILE = r'output\reference_characterize.npz'
    df = pd.read_csv(REFERENCE_FILE)
    # classIDx,frame_idx,mean_score,std_score
    class_ids = df['classIDx'].to_numpy()
    frame_ids = df['frame_idx'].to_numpy()
    scores_ids = df['mean_score'] - df['std_score']
    np.savez(
        OUTPUT_FILE,
        class_ids = class_ids,
        frame_ids = frame_ids,
        scores_ids = scores_ids,
    )

def get_maximum_index():

    REFERENCE_FILE = r'output\reference_characterization.csv'
    OUTPUT_FILE = r'output\keyframe_train_selection_1.csv'

    df = pd.read_csv(REFERENCE_FILE)
    
    data = []

    for i in range(0, 243):        
        df_video = df.loc[df['classIDx'] == i]
        dict_score = {}
        best_scores = []
        for idx, item in df_video.iterrows():
            score = item['mean_score'] - item['std_score']
            if score > -0.4:
                frame_id = item['frame_idx']
                dict_score[frame_id] = score
        
        max_index = int(max(dict_score, key = dict_score.get))
        best_scores.append(max_index)
        case = {
            'classIDx': i,
            'frame_idx': best_scores
        }

        data.append(case)
    

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(OUTPUT_FILE, index=False, header=True)

def analysic_error():

    REFERENCE_FILE = r'output\reference_characterization.csv'

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

    # print(data_list)
    x, y = zip(*data_list)

    plt.plot(x, y)
    plt.xlabel("Errors scores")
    plt.ylabel("Distribution")
    plt.show()

def check_high_std():

    REFERENCE_FILE = r'output\reference_characterization.csv'
    df = pd.read_csv(REFERENCE_FILE)

    x = np.arange(0., 0.13, 0.001)

    for i in x:
        th = np.round(i, 5)
        df_index = df.loc[df['std_score'] < th].copy()
        class_ids = df_index['classIDx'].to_list()
        df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
        index_list = df_index['period'].to_list()
        print(th, len(set(class_ids)), len(set(index_list)))

    # df_index = df.loc[df['std_score'] > 0.1]
    # for idx, item in df_index.iterrows():
    #     print(item['classIDx'], item['frame_idx'], item['std_score'])

def get_low_std():
    
    REFERENCE_FILE = r'output\reference_characterization.csv'
    OUTPUT_FILE = r'keyframe\reference_low_std.csv'
    df = pd.read_csv(REFERENCE_FILE)

    df_index = df.loc[df['std_score'] <= 0.05]

    df_index.to_csv(OUTPUT_FILE, index=False, header=True)   

    print("Number of references: {}".format(len(set(df_index['classIDx'].to_list()))))

def check_worst_cases():
    
    REFERENCE_FILE = r'keyframe\reference_low_std.csv'
    df = pd.read_csv(REFERENCE_FILE) 
   
    # x = np.arange(0., -0.45, -0.005)

    # for i in x:
    #     th = np.round(i, 5)
    #     df_index = df.loc[df['min_score'] > th].copy()
    #     class_ids = df_index['classIDx'].to_list()
    #     df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
    #     index_list = df_index['period'].to_list()
    #     print(th, len(set(class_ids)), len(set(index_list)))


    df_index = df.loc[df['max_score'] <= -0.3]
    for idx, item in df_index.iterrows():
        print(item['classIDx'], item['frame_idx'], item['max_score'])
    

def get_worst_valid():
    
    REFERENCE_FILE = r'keyframe\reference_low_std.csv'
    OUTPUT_FILE = r'keyframe\reference_worst_valid.csv'
    df = pd.read_csv(REFERENCE_FILE)

    df_index = df.loc[df['min_score'] >= -0.4]

    df_index.to_csv(OUTPUT_FILE, index=False, header=True)   

    print("Number of references: {}".format(len(set(df_index['classIDx'].to_list()))))

def check_full_separable():
    
    REFERENCE_FILE = r'keyframe\reference_low_std.csv'
    OUTPUT_FILE = r'iciap_data\KF_full_separable.csv'
    df = pd.read_csv(REFERENCE_FILE)    

    df_index = df.loc[df['min_score'] > 0.].copy()
    df_index.to_csv(OUTPUT_FILE, index=False, header=True)

    class_ids = df_index['classIDx'].to_list()
    df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
    index_list = df_index['period'].to_list()
    print(len(set(class_ids)), len(set(index_list)))

def get_full_separable():

    REFERENCE_FILE = r'keyframe\reference_worst_valid.csv'
    OUTPUT_FILE = r'keyframe\KF_full_separable.csv'
    REMAIN_FILE = r'keyframe\KF_remaining_full_separable.csv'

    df = pd.read_csv(REFERENCE_FILE)
    df_index = df.loc[df['min_score'] > 0.].copy() 
    class_ids = df_index['classIDx'].to_list()

    class_lst = set(class_ids)

    df_remained = df[~df.classIDx.isin(class_lst)]
    print(len(set(df_remained['classIDx'].to_list())))
    df_remained.to_csv(REMAIN_FILE, index=False, header=True)

    data = pd.DataFrame()

    for class_id in class_lst:
        df_tmp = df_index.loc[df_index['classIDx'] == class_id].copy()
        df_rs = df_tmp.loc[df_tmp['min_score'] == df_tmp['min_score'].max()]
        data = pd.concat([data, df_rs], ignore_index=True)


    data = data.sort_values(by=['classIDx'], ascending=True)
    
    # data.to_csv(OUTPUT_FILE, index=False, header=True)
   


def get_not_full_separable():
    
    REFERENCE_FILE = r'keyframe\reference_low_std.csv'
    OUTPUT_FILE =  r'keyframe\KF_not_separeble.csv'
    REMAIN_FILE = r'keyframe\KF_remaining_not_separeble.csv'

    df = pd.read_csv(REFERENCE_FILE)  

    df_index = df.loc[(df['min_score'] < 0.) & (df['max_score'] >= 0.)].copy()   

    class_ids = df_index['classIDx'].to_list()
    class_lst = set(class_ids)
    print(len(class_lst), len(df_index))
    df_remained = df[~df.classIDx.isin(class_lst)]
    # print(len(set(df_remained['classIDx'].to_list())))
    df_remained.to_csv(REMAIN_FILE, index=False, header=True)

    data = pd.DataFrame()

    for class_id in class_lst:
        df_tmp = df_index.loc[df_index['classIDx'] == class_id].copy()
        df_rs = df_tmp.loc[df_tmp['min_score'] == df_tmp['min_score'].max()]
        data = pd.concat([data, df_rs], ignore_index=True)


    data = data.sort_values(by=['classIDx'], ascending=True)
    
    # data.to_csv(OUTPUT_FILE, index=False, header=True)

    df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
    index_list = df_index['period'].to_list()
    df_index.to_csv(OUTPUT_FILE, index=False, header=True)
    print(len(set(class_ids)), len(set(index_list)))  
    # print(set(class_ids))

def get_not_separable():
    '''
    Not separable means low_std (std < alpha) and z_max <= 0 

    '''
    
    REFERENCE_FILE = r'keyframe\reference_low_std.csv'
    OUTPUT_FILE =  r'keyframe\reference_not_separable.csv'

    df = pd.read_csv(REFERENCE_FILE)
    class_ids = df['classIDx'].to_list()
    class_lst = set(class_ids)

    data = pd.DataFrame()

    mean_lst = []
    for class_id in class_lst:
        df_tmp = df.loc[df['classIDx'] == class_id].copy()
        df_ns = df_tmp.loc[df_tmp['max_score'] <= 0.].copy()
        if(len(df_ns) == 0):
            mean_avg = 0.
        else:
            mean_avg = np.mean(df_ns['mean_score'].to_numpy())
        # print(mean_avg)
        df_ns['normalized_mean'] =  np.round(df_ns['mean_score'] - mean_avg, 5)
        df_ns['mean_new'] =  np.round(mean_avg, 5)
        mean_lst.append(np.round(mean_avg, 5))
        df_ns['max_normalized'] =  np.round(df_ns['max_score'] - mean_avg, 5)
        data = pd.concat([data, df_ns], ignore_index=True)

    mean_lst.sort()
    mean_lst = mean_lst[:-5]
    print(mean_lst)
    print(np.min(mean_lst), np.max(mean_lst))

    x = np.arange(0, len(mean_lst), 1)
    plt.plot(x, mean_lst)
    plt.show()

    data = data.sort_values(by=['classIDx'], ascending=True)    
    # data.to_csv(OUTPUT_FILE, index=False, header=True)

def plot_not_separable():
    REFERENCE_FILE = r'keyframe\reference_not_separable.csv'
    OUTPUT_FILE =  r'keyframe\KF_NS.csv'

    df = pd.read_csv(REFERENCE_FILE)

    # df_tmp = df.loc[df['max_normalized'] < 0.].copy()

    df_tmp = df.loc[df['max_normalized'] >= 0.].copy()
    # print(len(df_tmp))
    # df_tmp.to_csv(OUTPUT_FILE, index=False, header=True)

    class_ids = df_tmp['classIDx'].to_list()
    class_lst = set(class_ids)

    print(len(df_tmp), len(class_lst))

    score_list = df['max_normalized'].to_numpy()

    x = np.arange(np.min(score_list), np.max(score_list), 0.01)

    data = {}
    
    for value in score_list:
        if value not in data:
            data[value] = 1
        else:
            data[value] += 1
        
        
    # for i in x:
    #     count = 0
    #     for value in score_list:
    #         if value < i:
    #             count += 1
    #     data[i] = count   

    data_list = sorted(data.items(), key=lambda x: x[0])
    # print(data_list)

    # print(data_list)
    x, y = zip(*data_list)

    plt.plot(x, y)
    # plt.vlines(x=0, ymin=0, ymax=14000)
    plt.xlabel("Errors scores")
    plt.ylabel("Distribution")
    plt.show()



def plot_std_worst():

    REFERENCE_FILE = r'keyframe\reference_characterization.csv'
    df = pd.read_csv(REFERENCE_FILE)

    std_list = np.arange(0.05, 0.13, 0.001)

       
    data_worst = {}
    data = {}
    for std_score in std_list:
        df_index = df.loc[(df['std_score'] > std_score) & (df['min_score'] <= -0.4)].copy()
        df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
        index_list = df_index['period'].to_list()
        value = len(set(index_list))
        data_worst[np.round(std_score, 3)] = value

        df_index = df.loc[(df['std_score'] > std_score) & (df['min_score'] > -0.4)].copy()
        df_index['period'] = df_index.classIDx.astype(str) + '-' + df_index.frame_idx.astype(str)
        index_list = df_index['period'].to_list()
        value = len(set(index_list))        
        data[np.round(std_score, 3)] = value
        

    # print(data)

    data_list = sorted(data_worst.items(), key=lambda x: x[0])
    data_lst = sorted(data.items(), key=lambda x: x[0])

    x, y = zip(*data_list)
    x_1, y_1 = zip(*data_lst)

    plt.plot(x, y, label= "high std, less discriminated")
    plt.plot(x_1, y_1, label= "high std, more discriminated")
    plt.xlabel("Standard deviation")
    plt.ylabel("Distribution")
    plt.legend(loc="upper right")
    plt.show()

def join_KF_files():

    FULL_SEPARABLE = r'keyframe\KF_full_separable.csv'
    NOT_SEPARABLE = r'keyframe\KF_not_separeble.csv'
    NOT_BAD = r'keyframe\KF_not_bad.csv'
    OUTPUT_FILE = r'keyframe\KF_one_per_reference.csv'

    df_s = pd.read_csv(FULL_SEPARABLE)
    df_n = pd.read_csv(NOT_SEPARABLE)
    df_b = pd.read_csv(NOT_BAD)

    df = pd.concat([df_s, df_n, df_b], ignore_index=True)

    df = df.sort_values(by=['classIDx'], ascending=True)
    print("Number of references: {}".format(len(set(df['classIDx'].to_list()))))
    df.to_csv(OUTPUT_FILE, index=False, header=True)


def main():
    print("[INFO] starting .........")
    since = time.time()

    # get_reference_duration()

    # plot_criterion()

    # plot_mean_std_error()

    # plot_error()

    # plot_index_error()

    # generate_reference_mean_std()

    # plot_reference()

    # join_sorted_score_reference_characterize()

    # save_reference_characterization()

    # analysic_error()

    # check_high_std()

    # get_low_std()

    # check_full_separable()
    # get_not_separable()
    plot_not_separable()

    # get_not_full_separable()
    

    # plot_std_worst()


    '''
    get 1 frame for 1 video
    '''
    # get_maximum_index()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))

if __name__ == '__main__':
    main()
    