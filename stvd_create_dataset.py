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

'''
training with siamese network by using Resnet feature
'''

class STVD_Retrain_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, train=False):
    
        self.img_dir = img_dir
        self.transform = transform  
        self.train = train

        if self.train:

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
        else:
            
            self.test_data_info = pd.read_csv(annotations_file)
            self.test_data = []

            for idx in self.test_data_info.index:
                classIDx = str(self.test_data_info.iloc[idx, 1])
                base_path = os.path.join(self.img_dir, classIDx)
                image_name = self.test_data_info.iloc[idx, 0]
                img_path = os.path.join(base_path, image_name)
                self.test_data.append(img_path)

            self.test_labels = np.asarray(self.test_data_info.iloc[:,1])
            self.test_labels = torch.from_numpy(self.test_labels)
            self.test_data_len = len(self.test_data_info.index)
        
    
    def __len__(self):
        if self.train:
            return self.train_data_len
        else:
            return self.test_data_len       

    def __getitem__(self, idx):

        if self.train:     
            img_path = self.train_data[idx]
            img_name = img_path.split('\\')[-1]
            img_name = img_name.split('.')[0]
            # image = Image.open(img_path)
            # if self.transform:
            #     image = self.transform(image)
            target = self.train_labels[idx] 
        else:
            img_path = self.test_data[idx]
            img_name = img_path.split('\\')[-1]
            img_name = img_name.split('.')[0]
            # image = Image.open(img_path)
            # if self.transform:
            #     image = self.transform(image)
            target = self.test_labels[idx] 
             
        return img_name, target

class Siamese_STVD_Dataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, stvd_dataset):

        self.stvd_dataset = stvd_dataset
        self.train = self.stvd_dataset.train
        self.transform = self.stvd_dataset.transform
        
        if self.train:
            self.train_labels = self.stvd_dataset.train_labels
            self.train_data = self.stvd_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            '''
            get label and coresponding indexes
            '''
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.stvd_dataset.test_labels
            self.test_data = self.stvd_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0] for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]), 1] for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i, random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.test_labels[i].item()])))]), 0]
                for i in range(1, len(self.test_data), 2)]
            
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()            
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]        
       
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), target

    def __len__(self):
        return len(self.stvd_dataset)



class Triplet_STVD_Dataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, stvd_dataset):
        self.stvd_dataset = stvd_dataset
        self.train = self.stvd_dataset.train
        self.transform = self.stvd_dataset.transform

        if self.train:
            self.train_labels = self.stvd_dataset.train_labels
            self.train_data = self.stvd_dataset.train_data

            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.stvd_dataset.test_labels
            self.test_data = self.stvd_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            
            self.test_triplets = triplets

    def __getitem__(self, index):

        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        # print(img1, img2, img3)
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        image3 = Image.open(img3)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        return (image1, image2, image3), []

    def __len__(self):
        return len(self.stvd_dataset)


# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from torchvision import transforms

# train_dataset= STVD_Feature_Dataset(    
#     annotations_file = r'training_data\train_descriptor.npz',    
#     train=True
# )

# val_dataset = STVD_Feature_Dataset(
#     annotations_file = r'training_data\val_descriptor.npz',
#     train=False,
# )

# siamese_train_dataset = Siamese_STVD_Dataset(train_dataset) 
# siamese_test_dataset = Siamese_STVD_Dataset(val_dataset)
# batch_size = 2
# # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)
# siamese_test_loader = DataLoader(siamese_test_dataset, shuffle=False)

# # Display image and label.
# (image_1, image_2), train_labels = next(iter(siamese_train_loader))
# print(f"Feature batch shape: {image_1.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = image_1[0].squeeze()
# label = train_labels[0]
# print(f"Label: {label}")