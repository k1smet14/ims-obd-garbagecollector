import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate import *
import numpy as np
from smp.segmentation_models_pytorch.losses.dice import DiceLoss
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
    
class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, classes=18, gamma=2.0, smoothing=0.0, weight=None):
        super(LabelSmoothingFocalLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.gamma = gamma
        self.cls = classes
        if weight is not None:
            self.weight = weight / weight.mean()

    def forward(self, pred, target):
        log_prob = pred.log_softmax(dim=-1)
        prob = torch.exp(log_prob)
        pred = ((1 - prob) ** self.gamma) * log_prob
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.weight is not None:
            return torch.mean(torch.sum(-true_dist * pred * self.weight, dim=-1))
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
    
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
    
class IoU_Focal_Loss(nn.Module):
    def __init__(self, iou_rate=0.4, weight=None, gamma=2.0, classes=12):
        nn.Module.__init__(self)
        self.n_class = classes
        self.rate = iou_rate
        self.focal = FocalLoss(weight=weight, gamma=gamma)

    def forward(self, preds, truth):
        focal = self.focal(preds, truth)
        preds = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
        miou = label_accuracy_score(truth.detach().cpu().numpy(), preds, n_class=self.n_class)
        return (1-miou)*self.rate + focal*(1-self.rate)
        
        
class IoU_CE_Loss(nn.Module):
    def __init__(self, iou_rate=0.4, weight=None, classes=12):
        nn.Module.__init__(self)
        self.n_class = classes
        self.rate = iou_rate
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds, truth):
        ce = self.ce(preds, truth)
        preds = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
        miou = label_accuracy_score(truth.detach().cpu().numpy(), preds, n_class=self.n_class)
        return (1-miou)*self.rate + ce*(1-self.rate)
    
    
class Dice_CE_Loss(nn.Module):
    def __init__(self, dice_rate=0.4, weight=None, classes=12):
        nn.Module.__init__(self)
        self.n_class = classes
        self.rate = dice_rate
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(mode='multiclass')

    def forward(self, preds, truth):
        ce = self.ce(preds, truth)
        dice = self.dice(preds, truth)
        return ce - torch.log(dice)