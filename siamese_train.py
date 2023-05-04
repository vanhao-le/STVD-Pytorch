import os
import time
import shutil
from tkinter.tix import Tree
import torch
import pandas as pd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable

from stvd_create_dataset import STVD_Retrain_Dataset, Siamese_STVD_Dataset

import chip.config as config
import matplotlib.pyplot as plt
from chip.siamese_train_model import train_model
from model.HAN import SiameseNet
from model.model_pooling import EmbeddingNet
from model.losses import ContrastiveLoss
from torchsummary import summary


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# torch.autograd.set_detect_anomaly(True)

def main():    
   
    image_w = config.image_w
    image_h = config.image_h
    image_mean = config.image_mean
    image_std = config.image_std
    annotation_train = config.annotation_train
    annotation_val = config.annotation_val  
    train_dir = config.image_train_dir
    val_dir = config.image_val_dir

    # Data augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_w, image_h)),            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomCrop((image_w, image_h)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_w, image_h)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]),
    }
    
    train_dataset= STVD_Retrain_Dataset(annotations_file = annotation_train, img_dir=train_dir, transform=data_transforms['train'], train=True)

    val_dataset= STVD_Retrain_Dataset(annotations_file = annotation_val, img_dir=val_dir, transform=data_transforms['val'], train=False)

    siamese_train_dataset = Siamese_STVD_Dataset(train_dataset)
    siamese_test_dataset = Siamese_STVD_Dataset(val_dataset)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siameses_train_loader = DataLoader(
        siamese_train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers = config.num_workers, pin_memory=True, drop_last=True
    )
    siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers = config.num_workers, pin_memory=True, drop_last=True
    )
       
    dataloaders= {'train': siameses_train_loader, 'val': siamese_test_loader}
    dataset_sizes = {'train': len(siameses_train_loader.dataset), 'val': len(siamese_test_loader.dataset)}
    print(dataset_sizes)
     
    margin = config.margin
    criterion = ContrastiveLoss(margin=margin)
    embedding_net = EmbeddingNet(pooling='rmac')  
    model = SiameseNet(embedding_net)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_of_gpus)  
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # device = 'cuda:0'

    model = model.to(device)

    # summary(embedding_net, (3, 192, 144))
    # return

  
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    
    momentum = config.momentum
    weight_decay = config.weight_decay
    step_size = config.step_size

    optimizer =  optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate,  momentum = momentum, weight_decay= weight_decay)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=-1)
    model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs)
    
    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']
    df.to_csv(r'training_data\siamese_log.csv',columns=['epoch','training_loss','val_loss'], header=True, index=False, encoding='utf-8')

    # fit(triplet_train_loader, triplet_test_loader, model, criterion, optimizer, exp_lr_scheduler, num_epochs, device, 1)


    model_save_filename = r'model_assets\siamese_model.pth'
    torch.save(model.state_dict(), model_save_filename)

    return



if __name__ == '__main__':
    main()
    
