import numpy as np
import pandas as pd

def extract_refernce_indexes():
    CATE_FILE = r"data\category.csv"
    KF_FILE = r'keyframe\static_fps_0000.csv'
    
    df = pd.read_csv(CATE_FILE)
    fps = 0.144
    # 1 frame / 7 seconds

    data = []
    for idx, item in df.iterrows():
        classIDx = item['classIDx']
        # duration_sec = max(int(item['duration_sec']), 1) 
        duration_sec = 1      
        ind_numpy = np.arange(1, duration_sec + 1, 1)
        ind_lst = ind_numpy[::7]
        case = {
            "classIDx": classIDx,
            "frame_idx": ind_lst.tolist()
        }
        data.append(case)

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(KF_FILE, index=False, header=True)


def extract_training_descriptor():

    KF_FILE = r'keyframe\static_fps_0000.csv'
    NEW_FILE = r'keyframe\fps_0000_train_descriptor.npz'

    POS_DESC = r'output\pos_train_descriptor.npz'
    query = np.load(POS_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']

    image_ids = []
    class_ids = []
    descriptors = []

    df = pd.read_csv(KF_FILE) 
    
    for idx, item in df.iterrows():
        r_classIDx = int(item['classIDx'])
        str_frame_lst = str(item['frame_idx'])
        frame_lst = np.fromstring(str_frame_lst[1:-1], dtype=np.int32, sep=',') 
        count =  0 
        # print(frame_lst)
        for i in range(rows):
            q_frame_idx = int(str(q_image_ids[i]).split('_')[-1])
            q_classIDx = int(q_class_ids[i])
            if q_classIDx == r_classIDx  and q_frame_idx in frame_lst:
                # print(q_frame_idx, q_classIDx)
                image_ids.append(q_image_ids[i])
                class_ids.append(q_class_ids[i])
                descriptors.append(q_descriptors[i])
                count += 1
        
        # print("class:", r_classIDx, "len:", count)



    print(len(image_ids))

    NEG_DESC = r'output\neg_train_descriptor.npz'
    query = np.load(NEG_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']

    for i in range(rows):
        image_ids.append(q_image_ids[i])
        class_ids.append(q_class_ids[i])
        descriptors.append(q_descriptors[i])

    print(len(image_ids))
    
    np.savez(
        NEW_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )



def extract_testing_descriptor():

    KF_FILE = r'keyframe\static_fps_0000.csv'

    NEW_FILE = r'keyframe\fps_0000_test_descriptor.npz'

    POS_DESC = r'output\pos_test_descriptor.npz'
    query = np.load(POS_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']

    image_ids = []
    class_ids = []
    descriptors = []

    df = pd.read_csv(KF_FILE)

    for idx, item in df.iterrows():
        r_classIDx = int(item['classIDx'])
        str_frame_lst = str(item['frame_idx'])
        frame_lst = np.fromstring(str_frame_lst[1:-1], dtype=np.int32, sep=',') 
        count =  0 
        # print(frame_lst)
        for i in range(rows):
            q_frame_idx = int(str(q_image_ids[i]).split('_')[-1])
            q_classIDx = int(q_class_ids[i])
            if q_classIDx == r_classIDx  and q_frame_idx in frame_lst:
                # print(q_frame_idx, q_classIDx)
                image_ids.append(q_image_ids[i])
                class_ids.append(q_class_ids[i])
                descriptors.append(q_descriptors[i])
                count += 1
        
        # print("class:", r_classIDx, "len:", count)  
    
    
    print(len(image_ids))
    NEG_DESC = r'output\neg_test_descriptor.npz'
    query = np.load(NEG_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']

    for i in range(rows):
        image_ids.append(q_image_ids[i])
        class_ids.append(q_class_ids[i])
        descriptors.append(q_descriptors[i])

    print(len(image_ids))
    
    np.savez(
        NEW_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )

def main():

    extract_refernce_indexes()

    extract_training_descriptor()
    extract_testing_descriptor()



if __name__ == '__main__':
    main()
    
