import torch
import torch.nn as nn
import model.functional as LF
from torch.nn.parameter import Parameter


class MAC(nn.Module):
    
    def __init__(self):
        super(MAC,self).__init__()

    def forward(self, x):

        return LF.mac(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'



class SPoC(nn.Module):
    
    def __init__(self):
        super(SPoC,self).__init__()

    def forward(self, x):
        return LF.spoc(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class RMAC(nn.Module):
    
    def __init__(self, L=3, eps=1e-6):
        super(RMAC,self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

class GeM(nn.Module):
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        # self.eps = torch.tensor(eps)

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=1e-6)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'