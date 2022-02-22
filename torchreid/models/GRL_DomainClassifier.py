import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


__all__ = ['grl']


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, delta):
        ctx.delta = delta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.delta
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        x = torch.sigmoid(x)
        return x

class DANN_GRL_model(nn.Module):
    def __init__(self):
        super(DANN_GRL_model, self).__init__()

        self.domain_classifier = Discriminator(input_dim=2048, hidden_dim=4096)

    def forward(self, input, delta=1, source=True):

        x = input.view(input.size(0), -1)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)

        return domain_pred


def grl(num_classes, loss, pretrained, use_gpu):
        return DANN_GRL_model()
