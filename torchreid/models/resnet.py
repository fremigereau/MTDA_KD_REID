"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import

import copy

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import math
from scipy.stats import norm

__all__ = ['resnet18','resnet34','resnet50','resnet101','resnet152']

model_urls = {'resnet18':'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34':'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50':'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101':'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152':'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
              }



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64'
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.
    
    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        fc_dim=2048,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,  # was 2 initially
        dropout_p=None,
        teacher_arch=None,
        **kwargs
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.teacher_arch = teacher_arch
        self.margins = None
        self.out_dim = 512 * block.expansion
        self.feature_dim = self.out_dim
        self.fc_dim = fc_dim
        self.inplanes = 64
        self.dilation = 1
        self.expansion = block.expansion
        self.multi_head = False
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".
                format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=last_stride,
            dilate=replace_stride_with_dilation[2]
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if fc_dim > 0:
            self.feat = nn.Linear(self.out_dim, self.feature_dim)
            self.feat_bn = nn.BatchNorm1d(self.feature_dim)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            self.feature_dim = fc_dim

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

        if self.teacher_arch != None:
            if self.teacher_arch == "resnet50" or self.teacher_arch == "resnet101" or self.teacher_arch == "resnet152":
                teacher_feat_dims = [256, 512, 1024, 2048]
            else:
                teacher_feat_dims = [64, 128, 256, 512]
            student_feat_dims = [64 * self.expansion, 128 * self.expansion, 256 * self.expansion,
                                 512 * self.expansion]
            # 1x1 conv to match smaller resnet feature dimension with larger models
            if self.loss == 'kd_reid':
                self.feat_matcher_list = nn.ModuleList([self._construct_feat_matchers(s, t) for s, t in zip(student_feat_dims, teacher_feat_dims)])


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


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


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def get_margin_from_bn(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise KeyError('ResNet unknown block error !!!')

        bns = [bn1, bn2, bn3, bn4]

        for i, bn in enumerate(bns):
            margin = []
            std = bn.weight.data
            mean = bn.bias.data
            for (s, m) in zip(std, mean):
                s = abs(s.item())
                m = m.item()
                if norm.cdf(-m / s) > 0.001:
                    margin.append(
                        - s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
                else:
                    margin.append(-3 * s)
            margin = torch.FloatTensor(margin).to(std.device)
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())
        return margin

    def get_channel_num(self):

        return [64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]


    def forward(self, input, target=None):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f = F.relu(f4)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc_dim > 0:
            if self.multi_head:
                v = self.feat_fc_multi[target](v)
            else:
                v = self.feat_bn(self.feat(v))

        if not self.training:
            v = F.normalize(v)
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'kd_mmd' or self.loss == 'mmd' or self.loss == 'triplet':
            return y, v
        elif self.loss == 'kd_reid':
            # Margin ReLU if teacher, 1x1 Conv for student
            if self.teacher_arch == None:
                f1 = torch.max(f1, getattr(self, 'margin%d' % (1)))
                f1 = f1.view(f1.size(0), -1)
                f2 = torch.max(f2, getattr(self, 'margin%d' % (2)))
                f2 = f2.view(f2.size(0), -1)
                f3 = torch.max(f3, getattr(self, 'margin%d' % (3)))
                f3 = f3.view(f3.size(0), -1)
                f4 = torch.max(f4, getattr(self, 'margin%d' % (4)))
                f4 = f4.view(f4.size(0), -1)
            else:
                f1 = self.feat_matcher_list[0](f1)
                f1 = f1.view(f1.size(0), -1)
                f2 = self.feat_matcher_list[1](f2)
                f2 = f2.view(f2.size(0), -1)
                f3 = self.feat_matcher_list[2](f3)
                f3 = f3.view(f3.size(0), -1)
                f4 = self.feat_matcher_list[3](f4)
                f4 = f4.view(f4.size(0), -1)
            return [f1, f2, f3, f4], v, y

        elif self.loss == 'feat_kd':
            f1 = F.relu(f1)
            f1 = f1.view(f1.size(0), -1)
            f2 = F.relu(f2)
            f2 = f2.view(f2.size(0), -1)
            f3 = F.relu(f3)
            f3 = f3.view(f3.size(0), -1)
            f4 = F.relu(f4)
            f4 = f4.view(f4.size(0), -1)
            return [f1, f2, f3, f4], v, y
        elif self.loss == 'adv_feat_kd':
            f1 = F.relu(f1)
            f2 = F.relu(f2)
            f3 = F.relu(f3)
            f4 = F.relu(f4)
            return [f1, f2, f3, f4], v, y
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def convert_2_multi_head(model, multi_head):
    model.multi_head = True
    model.feat_fc_multi = nn.ModuleList()
    for t in range(multi_head):
        feat_tmp = copy.deepcopy(model.feat)
        feat_bn_tmp = copy.deepcopy(model.feat_bn)
        C = [feat_tmp, feat_bn_tmp]
        model.feat_fc_multi.append(nn.Sequential(*C))

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def resnet18(num_classes, loss='softmax', pretrained=True, teacher_arch=None, fc_dim=2048, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dim=fc_dim,
        dropout_p=None,
        teacher_arch=teacher_arch,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    model.margins = model.get_margin_from_bn()

    return model


def resnet34(num_classes, loss='softmax', pretrained=True, teacher_arch=None, fc_dim=2048, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dim=fc_dim,
        dropout_p=None,
        teacher_arch=teacher_arch,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    model.margins = model.get_margin_from_bn()

    return model

def resnet50(num_classes, loss='softmax', pretrained=True, teacher_arch=None, fc_dim=2048, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dim=fc_dim,
        dropout_p=None,
        teacher_arch=teacher_arch,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    model.margins = model.get_margin_from_bn()

    return model

def resnet101(num_classes, loss='softmax', pretrained=True, teacher_arch=None, fc_dim=2048, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dim=fc_dim,
        dropout_p=None,
        teacher_arch=teacher_arch,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    model.margins = model.get_margin_from_bn()

    return model

def resnet152(num_classes, loss='softmax', pretrained=True, teacher_arch=None, fc_dim=2048, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dim=fc_dim,
        dropout_p=None,
        teacher_arch=teacher_arch,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    model.margins = model.get_margin_from_bn()

    return model
