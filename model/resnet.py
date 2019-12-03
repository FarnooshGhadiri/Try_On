import torch
from torch import nn
from torchvision.models import resnet

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
        classes = self.Soft(self.sc(features,self.num_class))
        return classes









