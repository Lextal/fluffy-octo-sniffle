import torch
from torch import nn


def iou_scores(logits, y, n_class=19):
    """
    Computes unweighted IoU scores for multiclass segmentation

    :param logits: Variable or FloatTensor, assumed to be (N, C, H, W)
    :param y: Variable or LongTensor, assumed to be (N, H, W)
    :param n_class: number of classes
    """
    out = torch.max(logits, 1)[1]  # argmax
    res = []
    for cls in range(n_class):
        _p = (out == cls)
        _t = (y == cls)
        tp = (_p * _t).sum()
        union = ((_p + _t) > 0).sum()
        res.append(-1 if union == 0 else tp * 100 / union)
    return res


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

