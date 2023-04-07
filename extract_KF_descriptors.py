import numpy as np
import pandas as pd

def extract_training_descriptor():

    KF_FILE = r'output\peaked_train_matching_3.csv'

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
    df['period'] = df.classIDx.astype(str) + "-" + df.image_name.astype(str)
    for i in range(rows):
        str_cmp = str(q_class_ids[i]) + "-" + str(q_image_ids[i])
        if str_cmp in df['period'].values:
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

    NEW_FILE = r'output\KF_train_descriptor.npz'
    np.savez(
        NEW_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )



def extract_testing_descriptor():

    KF_FILE = r'output\keyframe_train_selection.csv'

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

    NEW_FILE = r'output\KF_test_descriptor.npz'
    np.savez(
        NEW_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )

def main():
    # extract_training_descriptor()

    extract_testing_descriptor()

if __name__ == '__main__':
    main()
    
