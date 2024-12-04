import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn

from arch.preprocess import DataNormalizeLayer
from arch.resnet_svhn import SVHN_ResNet
from arch.resnet_cifar10 import CIFAR10_ResNet
from arch.wrn_cifar10 import WideResNet

svhn_normalize = {'bias': [0.5, 0.5, 0.5], 'scale': [0.5, 0.5, 0.5]}
cifar10_normalize = {'bias': [0.4914, 0.4822, 0.4465], 'scale': [0.2023, 0.1994, 0.2010]}

def parse_model(dataset, model_type, normalize = None, **kwargs):

    if isinstance(normalize, str):
        if normalize.lower() in ['cifar10', 'cifar10_normalize',]:
            normalize_layer = DataNormalizeLayer(bias = cifar10_normalize['bias'], scale = cifar10_normalize['scale'])
        elif normalize.lower() in ['svhn',]:
            normalize_layer = DataNormalizeLayer(bias = svhn_normalize['bias'], scale = svhn_normalize['scale'])
        else:
            raise ValueError('Unrecognized normalizer: %s' % normalize)
    elif normalize is not None:
        normalize_layer = DataNormalizeLayer(bias = normalize['bias'], scale = normalize['scale'])
    else:
        normalize_layer = DataNormalizeLayer(bias = 0., scale = 1.)

    if dataset.lower() in ['cifar10',]:
        if model_type.lower() in ['resnet',]:
            net = CIFAR10_ResNet(**kwargs)
        elif model_type.lower() in ['wide_resnet', 'wrn', 'wideresnet']:
            net = WideResNet(**kwargs)
        else:
            raise ValueError('Unrecognized architecture: %s' % model_type)
    elif dataset.lower() in ['svhn',]:
        if model_type.lower() in ['resnet',]:
            net = SVHN_ResNet(**kwargs)
        else:
            raise ValueError('Unrecognized architecture: %s' % model_type)
    else:
        raise ValueError('Unrecognized dataset: %s' % dataset)

    return nn.Sequential(normalize_layer, net)

def get_features(model, x, mode = None):

    x = model[0](x)
    x = model[1].get_features(x, mode)

    return x
