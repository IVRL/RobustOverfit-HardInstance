import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

## Tensor Operation
def group_add(x1_list, mul1, x2_list, mul2):
    '''
    >>> group summation: x1 * mul1 + x2 * mul2
    '''

    return [x1 * mul1 + x2 * mul2 for x1, x2 in zip(x1_list, x2_list)]

def group_product(x1_list, x2_list):
    '''
    >>> x1_list, x2_list: the list of tensors to be multiplied

    >>> group dot product
    '''

    return sum([torch.sum(x1 * x2) for x1, x2 in zip(x1_list, x2_list)])

def group_normalize(v_list):
    '''
    >>> normalize the tensor list to make them joint l2 norm be 1
    '''

    summation = group_product(v_list, v_list)
    summation = summation ** 0.5
    v_list = [v / (summation + 1e-6) for v in v_list]

    return v_list

def get_param(model):
    '''
    >>> return the parameter list
    '''
    param_list = []
    for param_name, param in model.named_parameters():
        param_list.append(param.data)
    return param_list

def get_param_grad(model):
    '''
    >>> return the parameter and gradient list
    '''
    param_list = []
    grad_list = []
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            param_list.append(param.data)
            grad_list.append(param.grad.data)
    return param_list, grad_list

## Model Operation
def distance_between_ckpts(ckpt1, ckpt2):
    '''
    >>> Calculate the distance ckpt2 - ckpt1
    '''

    assert len(ckpt1) == len(ckpt2), 'The length of ckpt1 should be the same as ckpt2'
    key_list = ckpt1.keys()

    distance_dict = OrderedDict()
    for key in key_list:
        param1 = ckpt1[key]
        param2 = ckpt2[key]
        distance_dict[key] = param2.data - param1.data

    return distance_dict

## Project to an adversarial budget
def project_(ori_pt, threshold, order = np.inf):

    if order in [np.inf,]:
        prj_pt = torch.clamp(ori_pt, min = - threshold, max = threshold) 
    elif order in [2,]:
        ori_shape = ori_pt.size()
        pt_norm = torch.norm(ori_pt.view(ori_shape[0], -1), dim = 1, p = 2)
        pt_norm_clip = torch.clamp(pt_norm, max = threshold)
        prj_pt = ori_pt.view(ori_shape[0], -1) / (pt_norm.view(-1, 1) + 1e-8) * (pt_norm_clip.view(-1, 1) + 1e-8)
        prj_pt = prj_pt.view(ori_shape)
    else:
        raise ValueError('Invalid norms: %s' % order)

    return prj_pt


def project(ori_pt, threshold, order = np.inf):
    '''
    Project the data into a norm ball

    >>> ori_pt: the original point
    >>> threshold: maximum norms allowed, can be float or list of float
    >>> order: norm used
    '''
    if isinstance(threshold, float):
        prj_pt = project_(ori_pt, threshold, order)
    elif isinstance(threshold, (list, tuple, np.ndarray)):
        if list(set(threshold)).__len__() == 1:
            prj_pt = project_(ori_pt, threshold[0], order)
        else:
            assert len(threshold) == ori_pt.size(0)
            prj_pt = torch.zeros_like(ori_pt)
            for idx, threshold_ in enumerate(threshold):
                prj_pt[idx: idx+1] = project_(ori_pt[idx: idx+1], threshold_, order)
    else:
        raise ValueError('Invalid value of threshold: %s' % threshold)

    return prj_pt

# For TRADES to search the adversarial budget
class TRADES_Criterion_V1(nn.Module):

    def __init__(self, fx):

        super(TRADES_Criterion, self).__init__()

        self.fx = fx.data
        self.logsoftmax_fx = F.log_softmax(fx, dim = -1).data

    def forward(self, logits, label_batch):

        prob = F.softmax(logits, dim = -1)
        loss_per_instance = - (self.logsoftmax_fx * prob).sum(dim = 1)

        return loss_per_instance.mean()

class TRADES_Criterion_V2(nn.Module):

    def __init__(self, fx):

        super(TRADES_Criterion_V2, self).__init__()

        self.softmax_fx = F.softmax(fx, dim = -1).data
        self.logsoftmax_fx = F.log_softmax(fx, dim = -1).data

    def forward(self, logits, label_batch):

        log_prob = F.log_softmax(logits, dim = -1)
        loss_per_instance = (self.softmax_fx * (self.logsoftmax_fx - log_prob)).sum(dim = 1)

        return loss_per_instance.mean()

class SemiAda_Criterion(nn.Module):

    def __init__(self, fx, gamma):

        super(SemiAda_Criterion, self).__init__()

        self.softmax_fx = F.softmax(fx, dim = -1).data
        self.gamma = gamma

    def forward(self, logits, label_batch):

        one_hot_label = torch.zeros_like(self.softmax_fx).scatter_(1, label_batch.view(-1, 1), 1)
        effect_label = one_hot_label * (1. - self.gamma) + self.softmax_fx * self.gamma
        loss_per_instance = (-F.log_softmax(logits, dim = -1) * effect_label).sum(dim = 1)

        return loss_per_instance.mean()
