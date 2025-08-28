"""
Advanced loss functions for better segmentation performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for imbalanced segmentation"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss combining focal and tversky"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        FocalTversky = (1 - Tversky)**self.gamma
        
        return FocalTversky


class IoULoss(nn.Module):
    """IoU Loss for direct IoU optimization"""
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Intersection and Union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - IoU


class ComboLoss(nn.Module):
    """Combination of multiple losses for optimal performance"""
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice Loss
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Cross entropy loss
        BCE = F.binary_cross_entropy(inputs, targets, reduce=False)
        
        # IoU Loss
        union = inputs.sum() + targets.sum() - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)
        
        # Combine losses
        combo = (self.ce_ratio * BCE) + ((1 - self.ce_ratio) * (1 - dice)) + (self.alpha * (1 - IoU))
        
        return combo.mean()


class AdvancedCombinedLoss(nn.Module):
    """Advanced combined loss with multiple components"""
    def __init__(self, weights=None):
        super(AdvancedCombinedLoss, self).__init__()
        if weights is None:
            weights = {
                'bce': 0.3,
                'focal': 0.2,
                'dice': 0.2,
                'tversky': 0.15,
                'iou': 0.15
            }
        self.weights = weights
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        self.iou_loss = IoULoss()

    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        
        if 'bce' in self.weights and self.weights['bce'] > 0:
            bce = self.bce_loss(predictions, targets)
            total_loss += self.weights['bce'] * bce
            loss_dict['bce'] = bce.item()
        
        if 'focal' in self.weights and self.weights['focal'] > 0:
            focal = self.focal_loss(predictions, targets)
            total_loss += self.weights['focal'] * focal
            loss_dict['focal'] = focal.item()
        
        if 'dice' in self.weights and self.weights['dice'] > 0:
            dice = self.dice_loss(predictions, targets)
            total_loss += self.weights['dice'] * dice
            loss_dict['dice'] = dice.item()
        
        if 'tversky' in self.weights and self.weights['tversky'] > 0:
            tversky = self.tversky_loss(predictions, targets)
            total_loss += self.weights['tversky'] * tversky
            loss_dict['tversky'] = tversky.item()
        
        if 'iou' in self.weights and self.weights['iou'] > 0:
            iou = self.iou_loss(predictions, targets)
            total_loss += self.weights['iou'] * iou
            loss_dict['iou'] = iou.item()
        
        return total_loss


class DiceLoss(nn.Module):
    """Standard Dice Loss"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
