import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Focal Loss
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
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


# LabelSmoothingLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
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


# F-1 Loss
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


# cross entropy + IoU
def fast_hist2(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score2(hist):
    """Returns accuracy score evaluation result.
      - mean IU
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)

    return mean_iu

class IoU_CE_Loss(nn.Module):
    def __init__(self, iou_rate=0.4, weight=None, classes=12):
        nn.Module.__init__(self)
        self.n_class = classes
        self.rate = iou_rate
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds, truth):
        ce = self.ce(preds, truth)
        preds = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
        miou = label_accuracy_score2(fast_hist2(truth.detach().cpu().numpy(), preds, n_class=self.n_class))
        return (1-miou)*self.rate + ce*(1-self.rate)


# _criterion_entrypoints = {
#     'cross_entropy': nn.CrossEntropyLoss,
#     'focal': FocalLoss,
#     'label_smoothing': LabelSmoothingLoss,
#     'f1': F1Loss
# }


# def criterion_entrypoint(criterion_name):
#     return _criterion_entrypoints[criterion_name]


# def is_criterion(criterion_name):
#     return criterion_name in _criterion_entrypoints


# def create_criterion(criterion_name, **kwargs):
#     if is_criterion(criterion_name):
#         create_fn = criterion_entrypoint(criterion_name)
#         criterion = create_fn(**kwargs)
#     else:
#         raise RuntimeError('Unknown loss (%s)' % criterion_name)
#     return criterion