from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(KLDivergenceLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, x, y):
        T = 20
        y_teacher = x
        y_student = y
        p = F.log_softmax(y_teacher/T, dim=1)
        q = F.softmax(y_student/T, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y_teacher.shape[0]

        return l_kl
