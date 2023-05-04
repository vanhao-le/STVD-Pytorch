import numpy as np
import pandas as pd

def extract_training_frames():

    KF_FILE = r'training_data\KF_one_per_reference.csv'
    NEW_FILE = r'training_data\train.csv'
    POS_DESC = r'training_data\train_positive_metadata.csv'

    df = pd.read_csv(KF_FILE)
    df_kf = pd.read_csv(POS_DESC)
    data = []   
    
    # image_name,classIDx
    for idx, item in df.iterrows():
        for idx1, k_item in df_kf.iterrows():
            r_classIDx = int(item['classIDx'])
            r_frameIDx = int(item['frame_idx'])
            q_frameIDx = int(str(k_item['image_name']).split('.')[0].split('_')[-1])
            q_classIDx = int(k_item['classIDx'])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:
                data.append(k_item)               
    
    print(len(data))

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(NEW_FILE, index=False, header=True)


def extract_testing_frames():

    KF_FILE = r'training_data\KF_one_per_reference.csv'
    NEW_FILE = r'training_data\testing.csv'
    POS_DESC = r'training_data\test_positive_metadata.csv'

    df = pd.read_csv(KF_FILE)
    df_kf = pd.read_csv(POS_DESC)
    data = []   
    
    # image_name,classIDx
    for idx, item in df.iterrows():
        for idx1, k_item in df_kf.iterrows():
            r_classIDx = int(item['classIDx'])
            r_frameIDx = int(item['frame_idx'])
            q_frameIDx = int(str(k_item['image_name']).split('.')[0].split('_')[-1])
            q_classIDx = int(k_item['classIDx'])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:
                data.append(k_item)               
    
    print(len(data))

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(NEW_FILE, index=False, header=True)

def extract_val_frames():
    
    KF_FILE = r'training_data\KF_one_per_reference.csv'
    NEW_FILE = r'training_data\val.csv'
    POS_DESC = r'training_data\val_positive_metadata.csv'

    df = pd.read_csv(KF_FILE)
    df_kf = pd.read_csv(POS_DESC)
    data = []   
    
    # image_name,classIDx
    for idx, item in df.iterrows():
        for idx1, k_item in df_kf.iterrows():
            r_classIDx = int(item['classIDx'])
            r_frameIDx = int(item['frame_idx'])
            q_frameIDx = int(str(k_item['image_name']).split('.')[0].split('_')[-1])
            q_classIDx = int(k_item['classIDx'])
            if r_classIDx == q_classIDx and r_frameIDx == q_frameIDx:
                data.append(k_item)               
    
    print(len(data))

    df_rs = pd.DataFrame(data)
    df_rs.to_csv(NEW_FILE, index=False, header=True)

def main():
    extract_training_frames()
    extract_testing_frames()
    extract_val_frames()

if __name__ == '__main__':
    main()
    
