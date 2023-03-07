
import pandas as pd
from sklearn.utils import shuffle
import os
import numpy as np


DATA_PATH = r"data"
groundtruth_file = "negative_metadata.csv"
positive_path = os.path.join(DATA_PATH, groundtruth_file)
df = pd.read_csv(positive_path)

# suffle data before splitting
df = shuffle(df)


def split_by_fractions(df:pd.DataFrame, fracs:list, random_state:int=42):
    assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
    remain = df.index.copy().to_frame()
    res = []
    for i in range(len(fracs)):
        fractions_sum=sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain=remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]

train_ds, val_ds, test_ds = split_by_fractions(df, [0.6,0.2,0.2]) # e.g: [train, validation, test]

print(train_ds.shape, val_ds.shape, test_ds.shape)

# This shows that the dataframe has not been stratified correctly.
# using set() to remove duplicated from list
print("Number of unique labels in train: ", len(set(train_ds["classIDx"])))
print("Number of unique labels in val: ", len(set(val_ds["classIDx"])))
print("Number of unique labels in test: ", len(set(test_ds["classIDx"])))

train_ds.to_csv(r'data\neg_train_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")
val_ds.to_csv(r'data\neg_val_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")
test_ds.to_csv(r'data\neg_test_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")