import torch
from torch import nn
import torch.functional as F
from torchvision.models import resnet
from torch.nn import init
""""
class ResNetCelebA(nn.Module):
    def __init__(self,num_class=2):
        super(ResNetCelebA,self).__init__()
        net = resnet.resnet18()
        self.num_class = num_class
        self.base = torch.nn.Sequential(*list(net.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nf = 512
        self.fc = nn.Linear(self.nf,2)
        self.Soft = nn.Softmax()
    def forward(self, x):
        features = self.base(x)
        features = self.avg_pool(features)
        features = features.view(-1, self.nf)
        classes = self.Soft(self.fc(features,self.num_class))
        return classes
"""
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
class ResNet(nn.module):

    __factory = {

        18 : resnet.resnet18(),
        34 : resnet.resnet34(),
        50 : resnet.resnet50(),
        101 : resnet.resnet101(),
        152 :resnet.resnet152(),

    }
    def __init__(self, depth, pretrain =True,cut_at_pooling=False,num_features=0,
                   norm=False,dropout=0,num_class=0):
        super(ResNet,self).__init__()
        self.depth = depth
        self.pretrain = pretrain
        self.cut_at_pooling = cut_at_pooling
        if depth not in ResNet.__factory:
             raise KeyError("unsupported depth",depth)
        self.base = ResNet.__factory[depth](pretrain)

        if not cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_class = num_class
            #append new layers
            out_features = self.base.fc.in_fetures()

            if self.has_embedding():
                self.feat = nn.Linear(out_features,self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight,mode='fan_out')
                init.constant(self.feat.bias)
                init.constant(self.feat_bn.weight,1)
                init.constant(self.feat_bn.bias,0)
            else:
                #change the num_features to CNN output channel
                self.num_features = out_features
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_class > 0:
                self.classifier = nn.linear(self.num_features,self.num_class)
                init.normal(self.classifier.weight,std =0.01)
                init.constant(self.classifier.bias,0)
        if not self.pretrain:
            self.reset_params()

    def forward(self,x):
        for name,module in self.base._module.items:
            if name == 'avgpool':
                break
        x = module(x)
        x = F.avg_pool2d(x,x.size()[2:])
        x = x.view(x.size(0), -1)
        if self.cut_at_pooling:
            return x
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.classifier > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules:
            if isinstance(m,nn.conv2d):
                init.kaiming_normal(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias,0)

                init.constant(m.bias)
                init.constant(m.batch)
            elif isinstance(m,nn.BatchNorm2d):
                init.constant(m.weight,1)
                init.constant(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal(m.weight,std=0.01)
                if m.bias is not None:
                    init.constant(m.bias,0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
























