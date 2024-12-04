import os
import sys
sys.path.insert(0, './')

import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack.autopgd_base import APGDAttack
from autoattack.fab_base import FABAttack
from autoattack.square import SquareAttack

from .utility import project

h_message = '''
>>> PGD(step_size, threshold, iter_num, order = np.inf)
>>> APGD(threshold, iter_num, rho, loss_type, alpha = 0.75, order = np.inf)
>>> Square(threshold, window_size_factor, iter_num, order = np.inf)
'''

def parse_attacker(name, mode = 'test', **kwargs):

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['pgd',]:
        attacker = PGD(**kwargs)
        if mode.lower() in ['test',]:
            attacker.adaptive = False
            attacker.use_worst = True
            attacker.max_loss = np.inf
            print('In test mode, adaptive is disabled, use_worst is enabled.')
        return attacker
    elif name.lower() in ['apgd',]:
        return APGD(**kwargs)
    elif name.lower() in ['fab',]:
        return FAB(**kwargs)
    elif name.lower() in ['square',]:
        return Square(**kwargs)
    else:
        raise ValueError('Unrecognized name of the attacker: %s' % name)

class DLRLoss(nn.Module):

    def __init__(self,):

        super(DLRLoss, self).__init__()

    def forward(self, x, y):

        x_sorted, ind_sorted = x.sort(dim = 1)
        ind = (ind_sorted[:, -1] == y).float()

        dividend = x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
        divisor = x_sorted[:, -1] - x_sorted[:, -3] + 1e-12

        return - (dividend / divisor).mean()

class PGD(object):

    def __init__(self, step_size, threshold, iter_num, loss_type = 'ce', adaptive = 0, use_worst = 1, order = np.inf, max_loss = -1, log = 0):

        self.step_size = step_size if step_size < 1. else step_size / 255.
        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.step_size_threshold_ratio = self.step_size / (self.threshold + 1e-6)
        self.iter_num = int(iter_num)
        self.adaptive = True if int(adaptive) != 0 else False
        self.use_worst = True if int(use_worst) != 0 else False
        self.max_loss = np.inf if max_loss <= 0 else max_loss
        self.order = order if order > 0 else np.inf
        self.loss_type = loss_type

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        self.log = False if int(log) == 0 else True

        self.status = 'normal'

        self.use_trace = False

        if self.loss_type.lower() in ['ce',]:
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type.lower() in ['dlr',]:
            self.criterion = DLRLoss()
        else:
            raise ValueError('Invalid loss type: %s' % self.loss_type)

        print('Create a PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, order = %f, use_worst = %s' % (
            self.step_size, self.threshold, self.iter_num, self.order, self.use_worst))
        print('loss_type = %s' % self.loss_type)

    def adjust_threshold(self, threshold, log = True):

        if isinstance(threshold, (list, tuple)):
            self.threshold = threshold
            self.step_size = [threshold_ * self.step_size_threshold_ratio for threshold_ in self.threshold]
        else:
            threshold = threshold if threshold < 1. else threshold / 255.

            self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
            self.threshold = threshold

            if log == True:
                print('Attacker adjusted, threshold = %1.2e, step_size = %1.2e' % (self.threshold, self.step_size))

    def enable_warmup(self, log = True):

        if self.status == 'normal':
            self.adaptive_cached = self.adaptive
            self.use_worst_cached = self.use_worst
            self.adaptive = False
            self.use_worst = True
            self.status = 'warmup'
            if log == True:
                print('Warmup is enabled, adaptive = %s, use_worst = %s' % (self.adaptive, self.use_worst))

    def disable_warmup(self, log = True):

        if self.status == 'warmup':
            self.adaptive = self.adaptive_cached
            self.use_worst = self.use_worst_cached
            self.status = 'normal'
            if log == True:
                print('Warmup is disabled, adaptive = %s, use_worst = %s' % (self.adaptive, self.use_worst))

    def attack(self, model, data_batch, label_batch, criterion = None):

        data_batch = data_batch.detach()
        label_batch = label_batch.detach()
        device = data_batch.device

        batch_size = data_batch.size(0)
        step_size = self.step_size if isinstance(self.step_size, (list, tuple)) else [self.step_size for _ in range(batch_size)]
        threshold = self.threshold if isinstance(self.threshold, (list, tuple)) else [self.threshold for _ in range(batch_size)]
        step_size_pt = torch.FloatTensor(step_size).to(device)
        threshold_pt = torch.FloatTensor(threshold).to(device)
        broadcast_shape = [-1,] + [1,] * (data_batch.dim() - 1)

        criterion = self.criterion.cuda(device) if criterion is None else criterion.cuda(device)

        if np.max(threshold) < 1e-6:
            return data_batch, label_batch

        ori_batch = data_batch.detach()

        # Initial perturbation
        noise = project(ori_pt = (torch.rand_like(data_batch) * 2 - 1) * threshold_pt.view(*broadcast_shape), threshold = np.abs(threshold), order = self.order)
        data_batch = torch.clamp(data_batch + noise, min = 0., max = 1.)
        data_batch = data_batch.detach().requires_grad_()

        worst_loss_per_instance = None
        correctness_bits = [1, ] * data_batch.size(0)
        weak_adv_batch = torch.zeros_like(data_batch)
        adv_data_batch = deepcopy(data_batch.detach())

        prediction_trajectory = []
        adv_data_batch_trajectory = []

        for iter_idx in range(self.iter_num):

            logits = model(data_batch)
            loss = criterion(logits, label_batch)
            _, prediction = logits.max(dim = 1)
            indicator_vec = (prediction == label_batch).int().data.cpu().numpy()

            prediction_trajectory.append(prediction.data.cpu().numpy())
            adv_data_batch_trajectory.append(deepcopy(data_batch.data))

            # Update the weak adversarial cases
            for instance_idx, (correctness_bit, indicator_bit) in enumerate(zip(correctness_bits, indicator_vec)):
                if np.abs(correctness_bit - 1) < 1e-6 and np.abs(indicator_bit) < 1e-6:
                    weak_adv_batch[instance_idx] = deepcopy(data_batch[instance_idx].detach())
                correctness_bits[instance_idx] = indicator_bit

            # Gradient
            loss.backward()
            grad = data_batch.grad.data

            # Update worst loss
            loss_per_instance = - F.log_softmax(logits).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if worst_loss_per_instance is None:
                worst_loss_per_instance = - torch.ones_like(loss_per_instance) * np.inf
            update_frozen_bit = (loss_per_instance > self.max_loss).float()

            loss_update_bits = (loss_per_instance * torch.sign(step_size_pt) >= worst_loss_per_instance).float()
            mask_shape = [-1,] + [1,] * (data_batch.dim() - 1)
            loss_update_bits = loss_update_bits.view(*mask_shape)
            adv_data_batch = deepcopy((adv_data_batch * (1. - loss_update_bits) + data_batch * loss_update_bits).detach())
            worst_loss_per_instance = torch.max(worst_loss_per_instance, loss_per_instance * torch.sign(step_size_pt))

            if self.order == np.inf:
                next_point = data_batch + torch.sign(grad) * step_size_pt.view(*broadcast_shape)
            elif self.order == 2:
                ori_shape = data_batch.size()
                grad_norm = torch.norm(grad.view(ori_shape[0], -1), dim = 1, p = 2)
                perb = (grad.view(ori_shape[0], -1) + 1e-8) / (grad_norm.view(-1, 1) + 1e-8) * step_size_pt.view(-1, 1)
                next_point = data_batch + perb.view(ori_shape)
            else:
                raise ValueError('Invalid norm: %s' % str(self.order))

            next_point = ori_batch + project(ori_pt = next_point - ori_batch, threshold = np.abs(threshold), order = self.order)
            next_point = torch.clamp(next_point, min = 0., max = 1.)

            data_batch = (data_batch * update_frozen_bit.view(*mask_shape) + next_point * (1. - update_frozen_bit).view(*mask_shape)).detach().requires_grad_()

            model.zero_grad()

        if self.use_worst is True:
            if self.adaptive == True:
                for instance_idx, correctness_bit in enumerate(correctness_bits):
                    if np.abs(correctness_bit) < 1e-6:
                        adv_data_batch[instance_idx] = deepcopy(weak_adv_batch[instance_idx].detach())

            if self.log == False:
                return adv_data_batch, label_batch
            else:
                return adv_data_batch, label_batch, prediction_trajectory, adv_data_batch_trajectory
        else:
            if self.adaptive == True:
                for instance_idx, correctness_bit in enumerate(correctness_bits):
                    if np.abs(correctness_bit) < 1e-6:
                        data_batch[instance_idx] = deepcopy(weak_adv_batch[instance_idx].detach())

            if self.log == False:
                return data_batch, label_batch
            else:
                return data_batch, label_batch, prediction_trajectory, adv_data_batch_trajectory

class APGD(object):

    def __init__(self, threshold, iter_num, rho, loss_type = 'ce', alpha = 0.75, order = np.inf):

        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.step_size = self.threshold * 2
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf
        self.rho = rho
        self.alpha = alpha
        self.loss_type = loss_type

        self.status = 'normal'
        self.log = False

        self.meta_threshold = self.threshold
        self.meta_step_size = self.step_size

        print('Create a Auto-PGD attacker')
        print('step_size = %1.2e, threshold = %1.2e, iter_num = %d, rho = %.4f, alpha = %.4f, order = %f' % (
            self.step_size, self.threshold, self.iter_num, self.rho, self.alpha, self.order))
        print('loss type = %s' % self.loss_type)

    def adjust_threshold(self, threshold, log = True):

        threshold = threshold if threshold < 1. else threshold / 255.

        self.step_size = self.meta_step_size * threshold / (self.meta_threshold + 1e-6)
        self.threshold = threshold

        if log == True:
            print('Attacker adjusted, threshold = %1.2e, step_size = %1.2e' % (self.threshold, self.step_size))

    def enable_warmup(self, log = True):

        pass

    def disable_warmup(self, log = True):

        pass

    def attack(self, model, data_batch, label_batch, criterion):

        norm = {np.inf: 'Linf', 2: 'L2'}[self.order]

        attacker = APGDAttack(model, n_restarts = 5, n_iter = self.iter_num, verbose=False, eps = self.threshold,
            norm = norm, eot_iter = 1, rho = self.rho, seed = time.time(), loss = self.loss_type, device = data_batch.device)

        _, adv_data_batch = attacker.perturb(data_batch, label_batch, cheap = True)

        return adv_data_batch.detach(), label_batch

class FAB(object):

    def __init__(self, threshold, iter_num, order = np.inf):

        self.order = order if order > 0 else np.inf
        self.threshold = threshold if threshold < 1. or self.order is not np.inf else threshold / 255.
        self.iter_num = int(iter_num)

        self.meta_threshold = self.threshold

        print('Create a FAB attacker')
        print('threshold = %1.2e, iter_num = %d, order = %s' %(
            self.threshold, self.iter_num, self.order))

    def adjust_threshold(self, threshold, log = True):

        threshold = threshold if threshold < 1. or self.order is not np.inf else threshold / 255.
        self.threshold = threshold

        if log == True:
            print('Attacker adjusted, threshold = %1.2e' % (self.threshold,))

    def attack(self, model, data_batch, label_batch, criterion = None, save_grad = False):

        norm = {np.inf: 'Linf', 2: 'L2'}[self.order]

        attacker = FABAttack(model, n_restarts = 5, n_iter = self.iter_num, eps = self.threshold,
                norm = norm, verbose = False, device = data_batch.device)

        adv_data_batch = attacker.perturb(data_batch, label_batch)

        if save_grad == True:
            adv_data_batch = adv_data_batch.requires_grad_()
            logits = model(adv_data_batch)
            loss = criterion(logits, label_batch)
            grad = torch.autograd.grad(loss, adv_data_batch, create_graph = True)[0]

            return adv_data_batch.detach(), label_batch, grad
        else:
            return adv_data_batch.detach(), label_batch

class Square(object):

    def __init__(self, threshold, window_size_factor, iter_num, order = np.inf):

        self.threshold = threshold if threshold < 1. else threshold / 255.
        self.window_size_factor = window_size_factor
        self.iter_num = int(iter_num)
        self.order = order if order > 0 else np.inf

        self.status = 'normal'
        self.log = False

        print('Create a Square attacker')
        print('threshold = %1.2e, window_size_factor = %d, iter_num = %d, order = %s' % (
            self.threshold, self.window_size_factor, self.iter_num, self.order))

    def adjust_threshold(self, threshold, log = True):

        threshold = threshold if threshold < 1. else threshold / 255.
        self.threshold = threshold
        
        if log == True:
            print('Attacker adjusted, threshold = %1.2e' % (self.threshold,))

    def enable_warmup(self, log = True):

        pass

    def disable_warmup(self, log = True):

        pass

    def attack(self, model, data_batch, label_batch, criterion):

        norm = {np.inf: 'Linf', 2: 'L2'}[self.order]

        attacker = SquareAttack(model, p_init = 0.8, n_queries = self.iter_num, eps = self.threshold, norm = norm,
            n_restarts = 1, seed = time.time(), verbose = False, device = data_batch.device, resc_schedule = False)

        adv_data_batch = attacker.perturb(data_batch, label_batch)

        return adv_data_batch.detach(), label_batch
