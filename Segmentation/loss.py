import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate import *
import numpy as np


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