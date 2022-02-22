from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from ..models.GRL_DomainClassifier import DANN_GRL_model
import torch.nn.functional as F

import numpy as np  #
from scipy.spatial import distance  #
from scipy.stats import norm  #
import matplotlib.pyplot as plt  #
import seaborn as sns  #
import pickle  #
import torch  #
from sklearn.cluster import KMeans #
import random
from torchreid.metrics import compute_distance_matrix

from functools import partial
from torch.autograd import Variable


class DomainClassifierLoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(DomainClassifierLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, teacher_features, grl_model, source=True, delta=1):

        domain_pred = grl_model(teacher_features, delta)

        if source:
            domain_label = torch.ones(len(domain_pred)).long().to(domain_pred.device).view(-1, 1)
        else:
            domain_label = torch.zeros(len(domain_pred)).long().to(domain_pred.device).view(-1, 1)

        loss_fn = nn.BCELoss()
        l_dc = loss_fn(domain_pred, domain_label.float())

        return l_dc

