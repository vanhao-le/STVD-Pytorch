import os
import numpy as np
import torch
import torch.nn as nn
from model.pooling import MAC, SPoC, RMAC, GeM
from model.normarlization import L2N
from torchsummary import summary
import torchvision.models as models
import torchvision

os.environ['TORCH_HOME'] = r"model_assets"

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'   : MAC,
    'spoc'  : SPoC,
    'rmac'  : RMAC,
    'gem'   : GeM,
}

class EmbeddingNet(nn.Module):    
    def __init__(self, pooling = 'rmac'):        
        super(EmbeddingNet, self).__init__()
        # net_in = models.resnet101(pretrained = True)
        # net_in = models.vgg16(pretrained = True)
        
        # net_in = models.inception_v3(pretrained = True) #Inception V3 results error for MAC extract

        net_in = models.googlenet(pretrained = True) #Inception V3 results error for MAC extract
        
        # summary(net_in, (3, 299, 299))

        for param in net_in.parameters():
            param.requires_grad = False

        
        '''
        initialize features take only convolutions for features, always ends with ReLU to make last activations non-negative
        '''
        features = list(net_in.children())[:-2] # -2 for ResNet; -1 for VGG
        # # features = list(net_in.features.children())[:-1]
        self.features = nn.Sequential(*features)
      
        self.pool = POOLING[pooling]()
        self.norm = L2N()

    def forward(self, input):
        # x -> features
        output = self.features(input)
        # features -> pool -> norm
        output = self.pool(output)
        output = self.norm(output).squeeze(-1).squeeze(-1)

        # it is Dx1 column vector per image (NxD if many images)        
        return output
    
    def get_embedding(self, x):
        return self.forward(x)
