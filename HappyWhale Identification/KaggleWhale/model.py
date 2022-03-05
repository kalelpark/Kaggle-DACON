import numpy as np
import torch
import torchvision.models as models
from torch.nn.modules.module import Module
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

def efficient_model(num_class = 0, model_name = 'b0'):
    if model_name in 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
    elif model_name in 'efficientnet-b1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
    elif model_name in 'efficientnet-b2':
        model = EfficientNet.from_pretrained('efficientnet-b2')
    elif model_name in 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
    elif model_name in 'efficientnet-b4':
        model = EfficientNet.from_pretrained('efficientnet-b4')
    elif model_name in 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
    elif model_name in 'efficientnet-b6':
        model = EfficientNet.from_pretrained('efficientnet-b6')
    elif model_name in 'efficientnet-b7':    
        model = EfficientNet.from_pretrained('efficientnet-b7')
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    model._fc = nn.Linear(model._fc.in_features, num_class, bias = True)

    return model

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

class CutMixCrossEntropyLoss(Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)
