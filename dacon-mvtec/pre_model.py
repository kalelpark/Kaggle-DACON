import timm
import torch
import torch.optim
import torch.nn as nn
from setting import config

class MvtecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(config.pre_trained_model, pretrained= True, num_classes = 88)
    
    def forward(self, x):
        x = self.model(x)
        return x