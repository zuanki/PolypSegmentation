import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss
    """

    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight, size_average=size_average)
        self.dice_loss = DiceLoss(mode='binary', from_logits=True)

    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        return bce_loss + dice_loss
