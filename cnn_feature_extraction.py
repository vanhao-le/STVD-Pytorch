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
from model.model_pooling import EmbeddingNet


torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = config.IMAGE_SIZE
annotations_file = r"data_setD\train_metadata.csv"
root_path = r"E:\STVD_DL\data_setD\train"
batch_size = config.BATCH_SIZE
os.environ['TORCH_HOME'] = r"model_assets"

OUTPUT_FILE = r'output_pooling\gglv1_train_descriptor.npz'

def get_vgg_model():    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.features = model.features[:]
    model.classifier = model.classifier[:4]
    model = model.eval()   
    return model

def get_resnet50_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    model.fc = torch.nn.Identity()
    model = model.eval()   
    return model

def get_inceptionv1_model():    
    model = models.googlenet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # print(model)
    model.fc = torch.nn.Identity()
    model = model.eval()   
    return model

def main():
    print("[INFO] starting .........")
    since = time.time()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    stvd_dataset =  STVDDataset(annotations_file=annotations_file, img_dir=root_path, transform=transform)
    stvd_loader = DataLoader(dataset=stvd_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # pretrained_model = get_resnet50_model()
    # pretrained_model = get_vgg_model()
    # print(pretrained_model)
    pretrained_model = get_inceptionv1_model()
    # print(pretrained_model)

    pool_names = ['mac', 'spoc', 'rmac', 'gem']
    # pretrained_model =  EmbeddingNet(pool_names[0])
    pretrained_model.eval()   
    # print(pretrained_model)
    
    # setting device on GPU if available, else CPU    
    print('Using device:', DEVICE)
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_of_gpus)  
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        pretrained_model = nn.DataParallel(pretrained_model)    

    # DEVICE = 'cuda:0'

    pretrained_model.to(DEVICE)
    summary(pretrained_model, (3, 299, 299))  

    # return

    image_ids = []
    class_ids = []
    descriptors = []

    with torch.no_grad():
        for img_names, images, labels in stvd_loader:            
            images = images.to(DEVICE)
            # outputs = pretrained_model(images)
            # for i in range(len(outputs)):                 
            #     outputs[i] = outputs[i] / (outputs[i].pow(2).sum(0, keepdim=True).sqrt())
            #     image_ids.append(str(img_names[i]))        
            #     class_ids.append(labels[i].numpy())
            #     descriptors.append(outputs[i].cpu().numpy().squeeze())
                # print(img_names[i], labels[i].numpy(), outputs[i].cpu().numpy().squeeze().shape)

            # break
            # synchronize the CUDA stream
            torch.cuda.synchronize()
    
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
    