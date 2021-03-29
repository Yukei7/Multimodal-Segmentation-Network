import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.eps = eps

    def forward(self, y_pred, y_true):
        num = y_true.size(0)
        probability = torch.sigmoid(y_pred)
        probability = probability.view(num, -1)
        targets = y_true.view(num, -1)
        assert (probability.shape == targets.shape)
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        return 1.0 - dice_score
