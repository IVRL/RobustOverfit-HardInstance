import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn

def calc_lip_linear(module, sample_input, iter_num = None):

    if iter_num is None:
        lip = torch.svd(module.weight)[1][0]
    else:
        rand_input = torch.randn_like(sample_input).requires_grad_()
        zero_input = torch.zeros_like(sample_input)
        for iter_num in range(iter_num):
            with torch.enable_grad():
                target = (module(rand_input) - module(zero_input)).view(rand_input.size(0), -1).norm(p = 2, dim = 1) ** 2 / 2
                vector = torch.autograd.grad(target.sum(), rand_input)[0]
                resize_shape = [rand_input.size(0),] + [1,] * (rand_input.dim() - 1)
                rand_input = vector / vector.view(rand_input.size(0), -1).norm(p = 2, dim = 1).view(*resize_shape)
                rand_input = rand_input.detach().requires_grad_()
        lip = (module(rand_input) - module(zero_input)).view(rand_input.size(0), -1).norm(p = 2, dim = 1)
        lip = torch.max(lip)
    return lip

def calc_lip_conv2d(module, sample_input, iter_num = None):

    assert iter_num is not None
    rand_input = torch.randn_like(sample_input).requires_grad_()
    zero_input = torch.zeros_like(sample_input)
    for iter_num in range(iter_num):
        with torch.enable_grad():
            target = (module(rand_input) - module(zero_input)).view(rand_input.size(0), -1).norm(p = 2, dim = 1) ** 2 / 2
            vector = torch.autograd.grad(target.sum(), rand_input)[0]
            resize_shape = [rand_input.size(0),] + [1,] * (rand_input.dim() - 1)
            rand_input = vector / vector.view(rand_input.size(0), -1).norm(p = 2, dim = 1).view(*resize_shape)
            rand_input = rand_input.detach().requires_grad_()
    lip = (module(rand_input) - module(zero_input)).view(rand_input.size(0), -1).norm(p = 2, dim = 1)
    lip = torch.max(lip)
    return lip
