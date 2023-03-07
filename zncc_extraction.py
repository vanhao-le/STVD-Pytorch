from torchvision import transforms
from torch import nn
import torch
from stvd_create_dataset import ZNCC_STVDDataset
from torch.utils.data import DataLoader
import chip.config as config
import torchvision.models as models
import os
import numpy as np
import time

annotations_file = r"data\train_metadata.csv"
root_path = r"E:\STVD_DL\data\train"
batch_size = config.BATCH_SIZE

OUTPUT_FILE = r'output\zncc_train_descriptor.npz'


def main():
    print("[INFO] starting .........")
    since = time.time()

    transform = transforms.Compose([        
        transforms.ToTensor(),        
    ])

    stvd_dataset =  ZNCC_STVDDataset(annotations_file=annotations_file, img_dir=root_path, img_w=80, img_h=60, transform=transform)
    stvd_loader = DataLoader(dataset=stvd_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
   
    image_ids = []
    class_ids = []
    descriptors = []

    with torch.no_grad():
        for img_names, images, labels in stvd_loader:
            for i in range(len(img_names)):
                # print(img_names[i], labels[i], outputs[i].shape)                 
                output = torch.flatten(images[i])                
                img_mean = torch.mean(output)
                img_std = torch.std(output)
                output = (output - img_mean) / img_std
                output = output.numpy().squeeze()
                # print(output)
                image_ids.append(str(img_names[i]))
                class_ids.append(labels[i].numpy())
                descriptors.append(output)

            # break
    
    np.savez(
        OUTPUT_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    )) 

if __name__ == '__main__':
    main()
    