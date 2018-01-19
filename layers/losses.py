import torch
from torch import nn


def dice(input, target):
    smooth = 1e-9

    iflat = input.view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class BCEWithDiceLoss(nn.Module):
    """
    Combines binary cross entropy loss with log-dice multiplied by alpha
    """
    
    def __init__(self, alpha=1.0):
        super(BCEWithDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, out, y):
        return self.bce(out, y) + self.alpha * dice(out, y)
