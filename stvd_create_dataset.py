import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torch

class STVDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):

        self.img_dir = img_dir
        self.transform = transform  

        self.train_data_info = pd.read_csv(annotations_file)
        self.train_data = []

        for idx in self.train_data_info.index:
            classIDx = str(self.train_data_info.iloc[idx, 1])
            base_path = os.path.join(self.img_dir, classIDx)
            image_name = self.train_data_info.iloc[idx, 0]
            img_path = os.path.join(base_path, image_name)
            self.train_data.append(img_path)

        self.train_labels = np.asarray(self.train_data_info.iloc[:,1])
        self.train_labels = torch.from_numpy(self.train_labels)
        self.train_data_len = len(self.train_data_info.index)
        
    
    def __len__(self):        
        return self.train_data_len        

    def __getitem__(self, idx):        
        img_path = self.train_data[idx]
        img_name = img_path.split('\\')[-1]
        img_name = img_name.split('.')[0]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        target = self.train_labels[idx]    
             
        return img_name, image, target

class ZNCC_STVDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_w=80, img_h=60, transform=None):

        self.img_dir = img_dir
        self.img_w = img_w
        self.img_h = img_h
        self.transform = transform  

        self.train_data_info = pd.read_csv(annotations_file)
        self.train_data = []

        for idx in self.train_data_info.index:
            classIDx = str(self.train_data_info.iloc[idx, 1])
            base_path = os.path.join(self.img_dir, classIDx)
            image_name = self.train_data_info.iloc[idx, 0]
            img_path = os.path.join(base_path, image_name)
            self.train_data.append(img_path)

        self.train_labels = np.asarray(self.train_data_info.iloc[:,1])
        self.train_labels = torch.from_numpy(self.train_labels)
        self.train_data_len = len(self.train_data_info.index)
        
    
    def __len__(self):        
        return self.train_data_len        

    def __getitem__(self, idx):        
        img_path = self.train_data[idx]
        img_name = img_path.split('\\')[-1]
        img_name = img_name.split('.')[0]
        image = Image.open(img_path).convert('L')
        image = image.resize((self.img_w, self.img_h))
        if self.transform:
            image = self.transform(image) 
        target = self.train_labels[idx]    
             
        return img_name, image, target