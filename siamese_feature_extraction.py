from torchvision import transforms
from torch import nn
import torch
from stvd_create_dataset import STVDDataset
from torch.utils.data import DataLoader
import chip.config as config
import torchvision.models as models
import os
import numpy as np
from torchsummary import summary
import time
from model.HAN import SiameseNet, TripletNet
from model.model_pooling import EmbeddingNet
from collections import OrderedDict

torch.backends.cudnn.benchmark = True

DEVICE = 'cuda:0'
image_w = config.image_w
image_h = config.image_h
image_mean = config.image_mean
image_std = config.image_std
annotations_file = r"training_data\testing.csv"
root_path = r"E:\STVD_DL\data\test"
batch_size = 256
os.environ['TORCH_HOME'] = r"model_assets"

OUTPUT_FILE = r'training_data\siamese_test_descriptor.npz'

MODEL_PATH = r'model_assets\siamese_model.pth'

def main():
    print("[INFO] starting .........")
    since = time.time()

    transform = transforms.Compose([
        transforms.Resize((image_w, image_h)),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])

    stvd_dataset =  STVDDataset(annotations_file=annotations_file, img_dir=root_path, transform=transform)
    stvd_loader = DataLoader(dataset=stvd_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embedding_net =  EmbeddingNet(pooling='rmac')
    model = SiameseNet(embedding_net)  
    

    # print(model)
    state_dict = torch.load(MODEL_PATH)    
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)    
    # params = model.state_dict()
    # print(params.keys())    

    model.eval()   
    
    model.to(DEVICE)

    # summary(pretrained_model, (3,224,224))

    # return

    image_ids = []
    class_ids = []
    descriptors = []

    with torch.no_grad():
        for img_names, images, labels in stvd_loader:            
            images = images.to(DEVICE)
            outputs = model.get_embedding(images)
            for i in range(len(outputs)):                 
                # outputs[i] = outputs[i] / (outputs[i].pow(2).sum(0, keepdim=True).sqrt())
                image_ids.append(str(img_names[i]))        
                class_ids.append(labels[i].numpy())
                descriptors.append(outputs[i].cpu().numpy().squeeze())
                # print(img_names[i], labels[i].numpy(), outputs[i].cpu().numpy().squeeze().shape)

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
    