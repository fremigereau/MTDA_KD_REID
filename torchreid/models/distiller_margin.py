from __future__ import division, absolute_import
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import nn
import math
from scipy.stats import norm

class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, delta):
            ctx.delta = delta
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.delta
            return output, None

def grad_reverse(x, delta=1):
    return GradReverse.apply(x, delta)

class Distiller_ADV(nn.Module):
    def __init__(self, t_net_list, s_net):
        super(Distiller_ADV, self).__init__()

        if torch.cuda.device_count() > 1:
            t_channels = t_net_list[0].module.get_channel_num()
            s_channels = s_net.module.get_channel_num()
        else:
            t_channels = t_net_list[0].get_channel_num()
            s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([self._construct_feat_matchers(s, t) for t, s in zip(t_channels, s_channels)])

        self.t_net_list = t_net_list
        self.s_net = s_net
        self.adv_conv1s = nn.ModuleList([self._construct_feat_matchers(t, 512) for t in t_channels])
        self.adv_conv2s = nn.ModuleList([self._construct_feat_matchers(512, 1) for _ in t_channels]) #nn.Conv2d(256, 1, kernel_size=1, stride=1)

    def _construct_feat_matchers(self, dim_in, dim_out):

        C = [nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(dim_out)]

        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return nn.Sequential(*C)

    def get_parameters(self):
        return list(self.adv_conv2s.parameters()) + list(self.Connectors.parameters()) + list(self.adv_conv1s.parameters())

    def forward(self, t_feats, s_feats):

        feat_num = len(t_feats)

        for i in range(feat_num):
            N, A, H, W = s_feats[i].shape
            s_feats[i] = grad_reverse(self.Connectors[i](s_feats[i]))
            #t_feats[i] = self.Connectors[i](t_feats[i])
            s_feats[i] = self.adv_conv1s[i](s_feats[i])
            t_feats[i] = self.adv_conv1s[i](t_feats[i])
            s_feats[i] = self.adv_conv2s[i](s_feats[i]).permute(0, 2, 3, 1).reshape(N, -1)
            t_feats[i] = self.adv_conv2s[i](t_feats[i]).permute(0, 2, 3, 1).reshape(N, -1)