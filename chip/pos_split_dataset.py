'''
This code used to split a dataset dedicated to the classification problem into three parts as train / val / test.
Train dataset: Set of data used for learning in order to fit the parameters to the machine learning model.
Valid dataset: Set of data used to provide an unbiased evaluation of a model fitted on the training dataset while tuning model hyperparameters.
In addition, it also plays a role in other forms of model preparation, such as feature selection, threshold cut-off selection.
Test dataset: Set of data used to provide an unbiased evaluation of a final model fitted on the training dataset.
Agrs: The csv input file formats as [video_name, class_name]
Return:
Three files train.csv / val.csv / test.csv
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

DATA_PATH = r"data"
groundtruth_file = "positive_metadata.csv"
positive_path = os.path.join(DATA_PATH, groundtruth_file)
df = pd.read_csv(positive_path)

# suffle data before splitting
df = shuffle(df)

# Let's say we want to split the data in 60/20/20 for train/valid/test dataset

data = df.drop(columns = ['classIDx']).copy()
labels = df['classIDx']

# In the first step we will split the data in training and remaining dataset
# Intent is that all labels should be present in both train and test.
x_train, x_remain, y_train, y_remain = train_test_split(data, labels, stratify=labels, train_size=0.6)

# Now since we want the valid and test size to be equal (20% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)

x_valid, x_test, y_valid, y_test = train_test_split(x_remain, y_remain, stratify=y_remain, test_size=0.5)

# print(x_train.shape), print(y_train.shape)
# print(x_valid.shape), print(y_valid.shape)
# print(x_test.shape), print(y_test.shape)

'''
Concatenate pandas objects along a particular axis.
axis{0/'index', 1/'columns'}, default 0
Example:
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['0', '1'])
s = pd.concat([s1, s2], axis= 1)
Result:
   0  1
0  a  0
1  b  1
'''

train_ds = pd.concat([x_train, y_train], axis=1)
val_ds = pd.concat([x_valid, y_valid], axis=1)
test_ds = pd.concat([x_test, y_test], axis=1)

# This shows that the dataframe has not been stratified correctly.
# using set() to remove duplicated from list
print("Number of unique labels in train: ", len(set(train_ds["classIDx"])))
print("Number of unique labels in val: ", len(set(val_ds["classIDx"])))
print("Number of unique labels in test: ", len(set(test_ds["classIDx"])))

train_ds.to_csv(r'data\pos_train_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")
val_ds.to_csv(r'data\pos_val_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")
test_ds.to_csv(r'data\pos_test_metadata.csv', columns=['video_name', 'classIDx', 'duration_sec', 'frame_per_sec', 'width', 'height'], index=False, header=True, encoding="utf-8-sig")