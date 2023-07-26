import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib 
from ast import literal_eval

def extract_positive_train_kf_descriptor():   
    
    KF_FILE = r'kf_character\KF_NC.csv'

    NEW_FILE = r'kf_character\KF_NC_pos_train_descriptor.csv'

    POS_DESC = r'output\pos_train_descriptor.npz'
    query = np.load(POS_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']   

    df = pd.read_csv(KF_FILE)
    print(rows)    

    data = []
    for idx, item in df.iterrows():
        r_classIDx = int(item['classIDx'])
        r_frameIDx = int(item['frame_idx'])
        for i in range(rows): 
            q_frameIDx = int(str(q_image_ids[i]).split('_')[-1])
            q_video = str(q_image_ids[i]).rsplit('_', 1)[0]
            q_classIDx = int(q_class_ids[i])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:                
                case = {
                    'class_ids': q_classIDx,
                    'video_ids': q_video, 
                    'image_ids': q_image_ids[i],
                    'frame_ids': q_frameIDx,
                    'descriptors': q_descriptors[i].tolist()
                }
                data.append(case)    
    
    df = pd.DataFrame(data)
    df.to_csv(NEW_FILE, index=False, header=True)    

def cosine_similarity():
    KF_FILE = r'kf_character\KF_Worst_pos_train_descriptor.csv'
    OUTPUT = r'kf_character\KF_Worst_score.csv'

    df = pd.read_csv(KF_FILE)

    class_lst = set(df['class_ids'].to_list())
    # print(class_lst)

    data = {}
    for class_id in class_lst:
        df_videos = df.loc[df['class_ids'] == class_id].copy()
        video_lst = set(df_videos['video_ids'])
        # print(video_lst)
        for video_id in video_lst:
            df_images = df_videos.loc[df_videos['video_ids'] == video_id].copy()
            sorted_df_images = df_images.sort_values(by=['frame_ids'], ascending=True).reset_index()
            # print(sorted_df_images[['class_ids', 'video_ids', 'image_ids', 'frame_ids']])
            image_num = len(sorted_df_images)
            if image_num  > 1:
                for i in range(0, image_num-2):                    
                    str_a = sorted_df_images.loc[i, 'descriptors']
                    str_b = sorted_df_images.loc[i+1, 'descriptors']
                    vec_a = np.array(literal_eval(str_a), dtype=object)
                    vec_b = np.array(literal_eval(str_b), dtype=object)
                    # print(vec_a.shape, vec_b.shape)
                    cosine_ab = np.round(np.dot(vec_a, vec_b), 5)
                    
                    if cosine_ab in data:
                        data[cosine_ab] += 1
                    else:
                        data[cosine_ab] = 1           

    lists = sorted(data.items()) # sorted by key, return a list of tuples

    df = pd.DataFrame(lists)
    df.to_csv(OUTPUT, index=False, header=False)
    x_pos, y_pos = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x_pos, y_pos)
    plt.show()   

def accumulate_cosine():

    KF_FILE = r'kf_character\KF_FS_score.csv'    

    df = pd.read_csv(KF_FILE)

    x = np.arange(0, 1.001, 0.001)

    data = {}
    for th in x:
        count = 0
        for idx, row in df.iterrows():
            if row['score'] <= th:
                count += int(row['occurrence'])
        
        data[th] = count


    lists = sorted(data.items())     
    x_pos, y_pos = zip(*lists) 
    plt.plot(x_pos, y_pos)
    plt.show() 


def main():
    # extract_positive_train_kf_descriptor()
    # cosine_similarity()

    accumulate_cosine()

    print()

if __name__ == '__main__':
    main()
    