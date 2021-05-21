# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
import torch
from my_utils import *


def validation(epoch, model, data_loader, criterion, device, n_class=12):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        # mIoU_list = []
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)
            # mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score2(hist)
        # mIoU_list.append(mIoU)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}, acc: {:.4f}, acc_cls: {:.4f}'.format(epoch, avrg_loss, mean_iu, acc, acc_cls))

    return avrg_loss, mean_iu


def validation3(epoch, model, data_loader, criterion, device, n_class=12):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((n_class, n_class))
        all_iou = []
        for step, (images, masks, _) in enumerate(data_loader):         

            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)  

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)
            mIoU_list.append(mIoU)
            
            batch_iou = batch_iou_score(masks.detach().cpu().numpy(), outputs, len(outputs))
            all_iou.append(batch_iou)
            
        avrg_loss = total_loss / cnt
        miou2 = mIoU_score(hist)
        miou3 = np.mean(all_iou)
        print('Validation #{}  Average Loss: {:.4f}, mIoU2: {:.4f}, mIOU3: {:.4f}'.format(epoch, avrg_loss, miou2, miou3))

    return avrg_loss, np.mean(mIoU_list), miou2, miou3


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=12):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    return mean_iu

def label_accuracy_score2(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def batch_iou_score(label_trues, label_preds, batch_size, n_class=12):
    hist = np.zeros((n_class, n_class))
    batch_iou = 0
    for lt, lp in zip(label_trues, label_preds):
        hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
            batch_iou += np.nanmean(iu) / batch_size
    return batch_iou


def mIoU_score(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu