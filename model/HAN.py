import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net   

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
    
    @torch.jit.export
    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    @torch.jit.export
    def get_embedding(self, x):
        return self.embedding_net(x)

# def test():
#     batch_size = 1
#     input_1 = Variable(torch.randn(batch_size, 3, 192, 144))
#     model = SiameseNet()
#     output1, output2 = model(input_1, input_1)
#     print(output1.shape, output2.shape)

# test()
