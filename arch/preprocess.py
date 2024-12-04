import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataNormalizeLayer(nn.Module):

    def __init__(self, bias, scale):

        super(DataNormalizeLayer, self).__init__()

        if isinstance(bias, float) and isinstance(scale, float):
            self._bias = torch.FloatTensor(1).fill_(bias).view(1, -1, 1, 1)
            self._scale = torch.FloatTensor(1).fill_(scale).view(1, -1, 1, 1)
        elif isinstance(bias, (tuple, list)) and isinstance(scale, (tuple, list)):
            self._bias = torch.FloatTensor(bias).view(1, -1, 1, 1)
            self._scale = torch.FloatTensor(scale).view(1, -1, 1, 1)
        else:
            raise ValueError('Invalid parameter: bias = %s, scale = %s' % (bias, scale))

    def forward(self, x):

        x = (x - self._bias.to(x.device)) / self._scale.to(x.device)

        return x

class DataMixLayer(nn.Module):

    def __init__(self, num_sources):

        super(DataMixLayer, self).__init__()

        self.num_sources = num_sources
        self.weight = nn.Parameter(torch.zeros([self.num_sources,]))


    def forward(self, x):

        assert len(x) == self.num_sources, 'The number of sources should be %d, %d found.' % (self.num_sources, len(x))

        relative_weights = self.weight - torch.max(self.weight)
        probability = torch.exp(relative_weights) / torch.sum(torch.exp(relative_weights))        

        return sum([probability[idx] * x[idx] for idx in range(self.num_sources)])
