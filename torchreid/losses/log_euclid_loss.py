from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid.metrics import compute_distance_matrix
import scipy.linalg

def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)

def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)

def logm(x):
    return Logm.apply(x)

class LogEuclidLoss(nn.Module):

    def __init__(self, use_gpu=True, log=False):
        super(LogEuclidLoss, self).__init__()
        self.use_gpu = use_gpu
        self.log = log

    def forward(self, feats_source, feats_target):

        dist_s = compute_distance_matrix(feats_source,feats_source,metric='cosine')
        dist_t = compute_distance_matrix(feats_target,feats_target,metric='cosine')

        if self.log:
            dist_s = logm(dist_s)
            dist_t = logm(dist_t)

        loss = torch.norm(dist_s - dist_t)
        return loss
