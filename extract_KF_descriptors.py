import numpy as np
import pandas as pd

def extract_training_descriptor():

    KF_FILE = r'iciap_data\KF_Worst.csv'

    NEW_FILE = r'iciap_data\KF_Worst_train_descriptor.npz'


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
    print(rows)
    # df['period'] = df.classIDx.astype(str) + "-" + df.image_name.astype(str)

    for idx, item in df.iterrows():
        r_classIDx = int(item['classIDx'])
        r_frameIDx = int(item['frame_idx'])
        for i in range(rows): 
            q_frameIDx = int(str(q_image_ids[i]).split('_')[-1])
            q_classIDx = int(q_class_ids[i])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:            
                image_ids.append(q_image_ids[i])
                class_ids.append(q_class_ids[i])
                descriptors.append(q_descriptors[i])
    
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

    KF_FILE = r'iciap_data\KF_Worst.csv'
    NEW_FILE = r'iciap_data\KF_Worst_test_descriptor.npz'    

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

    print(rows)
    # for idx, item in df.iterrows():
    #     r_classIDx = int(item['classIDx'])
    #     str_frame_lst = str(item['frame_idx'])
    #     frame_lst = np.fromstring(str_frame_lst[1:-1], dtype=np.int32, sep=',') 
    #     count =  0 
    #     # print(frame_lst)
    #     for i in range(rows):
    #         q_frame_idx = int(str(q_image_ids[i]).split('_')[-1])
    #         q_classIDx = int(q_class_ids[i])
    #         if q_classIDx == r_classIDx  and q_frame_idx in frame_lst:
    #             # print(q_frame_idx, q_classIDx)
    #             image_ids.append(q_image_ids[i])
    #             class_ids.append(q_class_ids[i])
    #             descriptors.append(q_descriptors[i])
    #             count += 1
        
        # print("class:", r_classIDx, "len:", count)
    
    for idx, item in df.iterrows():
        r_classIDx = int(item['classIDx'])
        r_frameIDx = int(item['frame_idx'])
        for i in range(rows): 
            q_frameIDx = int(str(q_image_ids[i]).split('_')[-1])
            q_classIDx = int(q_class_ids[i])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:            
                image_ids.append(q_image_ids[i])
                class_ids.append(q_class_ids[i])
                descriptors.append(q_descriptors[i])    
    
    
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

def extract_val_descriptor():
    
    KF_FILE = r'keyframe\KF_one_per_reference.csv'

    NEW_FILE = r'training_data\val_descriptor.npz'

    POS_DESC = r'output\pos_val_descriptor.npz'
    query = np.load(POS_DESC)
    rows = len(query['image_ids'])
    q_image_ids = query['image_ids']
    q_class_ids = query['class_ids']
    q_descriptors = query['descriptors']

    image_ids = []
    class_ids = []
    descriptors = []

    df = pd.read_csv(KF_FILE)
    print(rows)

    # for idx, item in df.iterrows():
    #     r_classIDx = int(item['classIDx'])
    #     str_frame_lst = str(item['frame_idx'])
    #     frame_lst = np.fromstring(str_frame_lst[1:-1], dtype=np.int32, sep=',') 
    #     count =  0 
    #     # print(frame_lst)
    #     for i in range(rows):
    #         q_frame_idx = int(str(q_image_ids[i]).split('_')[-1])
    #         q_classIDx = int(q_class_ids[i])
    #         if q_classIDx == r_classIDx  and q_frame_idx in frame_lst:
    #             # print(q_frame_idx, q_classIDx)
    #             image_ids.append(q_image_ids[i])
    #             class_ids.append(q_class_ids[i])
    #             descriptors.append(q_descriptors[i])
    #             count += 1
        
        # print("class:", r_classIDx, "len:", count)
    
    
    for i in range(rows):
        for idx, item in df.iterrows():
            r_classIDx = int(item['classIDx'])
            r_frameIDx = int(item['frame_idx'])
            q_frameIDx = int(str(q_image_ids[i]).split('_')[-1])
            q_classIDx = int(q_class_ids[i])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:            
                image_ids.append(q_image_ids[i])
                class_ids.append(q_class_ids[i])
                descriptors.append(q_descriptors[i])
    
    print(len(image_ids))

    # NEG_DESC = r'output\neg_test_descriptor.npz'
    # query = np.load(NEG_DESC)
    # rows = len(query['image_ids'])
    # q_image_ids = query['image_ids']
    # q_class_ids = query['class_ids']
    # q_descriptors = query['descriptors']

    # for i in range(rows):
    #     image_ids.append(q_image_ids[i])
    #     class_ids.append(q_class_ids[i])
    #     descriptors.append(q_descriptors[i])

    # print(len(image_ids))
    
    np.savez(
        NEW_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )

def main():
    extract_training_descriptor()
    extract_testing_descriptor()
    # extract_val_descriptor()

if __name__ == '__main__':
    main()
    
