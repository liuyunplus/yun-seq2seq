import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=None) -> None:
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predicted, target):
        ce_loss = self.loss_func(predicted, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss