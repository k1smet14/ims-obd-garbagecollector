import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class ResNet50(nn.Module):
    def __init__(self, num_classes=18, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    def forward(self, x):
        out = self.model(x)
        return out


class ResNet101(nn.Module):
    def __init__(self, num_classes=18, pretrained=True):
        super().__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    def forward(self, x):
        out = self.model(x)
        return out


class EfficientNet_b3(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)

    def forward(self, x):
        out = self.model(x)
        return out