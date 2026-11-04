# losses/dice_loss.py

import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, target):

        prediction = torch.sigmoid(logits)

        prediction = prediction.flatten(
            start_dim=1
        )

        target = target.flatten(
            start_dim=1
        )

        intersection = (
            prediction * target
        ).sum(dim=1)

        denominator = (
            prediction.sum(dim=1)
            + target.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()