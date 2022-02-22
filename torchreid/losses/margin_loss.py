from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(MarginLoss, self).__init__()
        self.use_gpu = use_gpu


    def forward(self, feats_source, feats_target):

        loss = nn.functional.mse_loss(feats_source, feats_target, reduction="none")
        loss = loss * ((feats_source > feats_target) | (feats_target > 0)).float()
        return loss.sum()
