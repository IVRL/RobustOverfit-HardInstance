import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.attack import parse_attacker
from util.lipschitz import calc_lip_linear, calc_lip_conv2d
from copy import deepcopy

class ResNet_Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride = 1):

        super(ResNet_Block, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.nonlinear1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.nonlinear2 = nn.ReLU(inplace = True)

        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size = 1, stride = self.stride, bias = False),
                nn.BatchNorm2d(self.out_planes)
                )

    def forward(self, x):

        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.nonlinear2(out)

        return out

    def calc_lip(self, x, iter_num = None):

        lip1 = calc_lip_conv2d(nn.Sequential(self.conv1, self.bn1), x, iter_num)
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        lip2 = calc_lip_conv2d(nn.Sequential(self.conv2, self.bn2), out, iter_num)
        lip_shortcut = calc_lip_conv2d(self.shortcut, x, iter_num)
        lip = lip1 * lip2 + lip_shortcut

        return lip

class CIFAR10_ResNet(nn.Module):

    def __init__(self, num_block_list = [2, 2, 2, 2], class_num = 10, **kwargs):

        super(CIFAR10_ResNet, self).__init__()

        self.num_block_list = num_block_list
        self.in_planes = 64
        self.class_num = class_num
        print('CIFAR10 ResNet: num_block_list = %s' % self.num_block_list)

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(out_planes = 64, num_blocks = num_block_list[0], stride = 1)
        self.layer2 = self._make_layer(out_planes = 128, num_blocks = num_block_list[1], stride = 2)
        self.layer3 = self._make_layer(out_planes = 256, num_blocks = num_block_list[2], stride = 2)
        self.layer4 = self._make_layer(out_planes = 512, num_blocks = num_block_list[3], stride = 2)

        self.classifier = nn.Linear(512, self.class_num)

    def _make_layer(self, out_planes, num_blocks, stride):

        stride_list = [stride,] + [1,] * (num_blocks - 1)

        transit_layer = ResNet_Block(in_planes = self.in_planes, out_planes = out_planes, stride = stride)
        layers = [transit_layer,]
        self.in_planes = out_planes
        for _ in range(num_blocks - 1):
            residual_layer = ResNet_Block(in_planes = self.in_planes, out_planes = out_planes, stride = 1)
            layers.append(residual_layer)

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.nonlinear1(self.bn1(self.conv1(x)))

        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        for layer in self.layer3:
            out = layer(out)
        for layer in self.layer4:
            out = layer(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(-1, 512)
        out = self.classifier(out)

        return out

    def calc_lip(self, x, iter_num = None):

        lip = calc_lip_conv2d(nn.Sequential(self.conv1, self.bn1), x, iter_num)
        out = self.nonlinear1(self.bn1(self.conv1(x)))

        for layer in self.layer1:
            lip = lip * calc_lip_conv2d(layer, out, iter_num)
            out = layer(out)
        for layer in self.layer2:
            lip = lip * calc_lip_conv2d(layer, out, iter_num)
            out = layer(out)
        for layer in self.layer3:
            lip = lip * calc_lip_conv2d(layer, out, iter_num)
            out = layer(out)
        for layer in self.layer4:
            lip = lip * calc_lip_conv2d(layer, out, iter_num)
            out = layer(out)

        lip = lip * calc_lip_conv2d(nn.Sequential(nn.AvgPool2d(4), nn.Flatten(), self.classifier), out, iter_num)

        return lip

    def get_features(self, x, mode = None):

        out = self.nonlinear1(self.bn1(self.conv1(x)))

        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        for layer in self.layer3:
            out = layer(out)
        for layer in self.layer4:
            out = layer(out)

        if mode is not None and mode.lower() in ['no_pool',]:
            return out
        
        out = F.avg_pool2d(out, 4)
        out = out.view(-1, 512)

        return out


