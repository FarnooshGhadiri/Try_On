import math
import torch
from torch import nn

class ElsWiseVer(nn.module):

    def __init__(self,nonlinearity = 'square', use_batch_norm=False,
                 use_classifer=False,num_features=0, num_classes=0 ):
        super(ElsWiseVer,self).__init__()
        self.nonlinearity = nonlinearity
        self.use_batch_norm = use_batch_norm
        self.use_classifer = use_classifer
        if nonlinearity is not None and nonlinearity not in ['square','abs']:
            raise KeyError("Not known nonlinearity")
        if self.use_batch_norm:
           self.bn = nn. BatchNorm1d(num_features)
           self.bn.weight.data.fill(1)
           self.bn.bias.data.zero_()
        if self.use_classifer:
            self.use_classifer = nn.Linear(num_features,num_classes)
            self.use_classifer.weight.data.normal_(0,0.001)
            self.use_classifer.bias.data.zero_()


    def forward(self,x1,x2): # x1 is a source or target images, x2 is x1's Clothes or x1's body parts
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        if self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifer:
            x = x.view(x.size(0),-1)
            x = self.use_classifer(x)
        else:
            x = x.sum(1)
        return x










