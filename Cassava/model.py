import timm
import torch
import torch.optim as optim
import torch.nn as nn
from setting import setting
config = setting

class CassavaNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(config.TIMM_MODEL, pretrained= True)
        n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]

        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward_features(self, x):
        x = self.backbone(x)
        return x
    
    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats


class SnapMixLosss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a*lam_a + loss_b * lam_b)
        return loss

