import os
import sys
sys.path.insert(0, './')

import json
import pickle
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.evaluation import *
from util.train import epoch_pass
from util.augmentation import apply_aug, load_delta, update_delta

def adv_train_fast(model, output_dim, train_loader, valid_loader, test_loader, aug_policy, threshold, step_size, loss_param, reweight, train_test,
    rho, beta, warmup, warmup_rw, delta_reset, test_attacker, epoch_num, epoch_ckpts, optimizer, lr_func, out_folder, model_name, device, criterion, tosave, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []
    valid_acc_last_epoch = 0.

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()
    attack_acc_calculator = AverageCalculator()
    attack_loss_calculator = AverageCalculator()

    idx2delta = {}
    idx2target = {}

    for epoch_idx in range(epoch_num):

        # Training phase
        model.train()
        batch_num = len(train_loader)
        if delta_reset is not None and epoch_idx % delta_reset == 0:
            idx2delta = {}
        acc_per_instance_this_epoch = {}
        loss_per_instance_this_epoch = {}
        entropy_per_instance_this_epoch = {}
        acc_calculator.reset()
        loss_calculator.reset()
        attack_acc_calculator.reset()
        attack_loss_calculator.reset()
        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

            # Update the learning rate
            epoch_batch_idx = epoch_idx + idx / batch_num
            if lr_func is not None:
                lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learning rate = %1.2e' % lr_this_batch)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            idx_batch = idx_batch.int().data.cpu().numpy()

            # Obtain delta
            if delta_reset is None:
                delta_batch = (torch.rand_like(data_batch) * 2 - 1) * threshold
                delta_batch = torch.clamp(data_batch + delta_batch, min = 0., max = 1.) - data_batch
                delta_batch = delta_batch.detach().requires_grad_()
            else:
                slices = []
                for idx_in_batch, instance_idx in enumerate(idx_batch):
                    if instance_idx.__int__() not in idx2delta:
                        if 'crop' in aug_policy:
                            channel_num, height, width = data_batch.size(1), data_batch.size(2), data_batch.size(3)
                            delta_this_instance = (torch.rand(channel_num, height + 8, height + 8).to(data_batch.device) * 2 - 1) * threshold
                        else:
                            delta_this_instance = (torch.rand_like(data_batch[idx_in_batch]) * 2 - 1) * threshold
                        slices.append(delta_this_instance.unsqueeze(0))
                        idx2delta[instance_idx.__int__()] = delta_this_instance
                    else:
                        slices.append(idx2delta[instance_idx.__int__()].unsqueeze(0))
                delta_batch = torch.cat(slices, dim = 0).detach().requires_grad_()

            # Apply augmentation
            ori_batch = data_batch.clone()
            aug_trans_log = None
            data_batch, soft_label, aug_trans_log = apply_aug(ori_batch, label_batch, output_dim, aug_policy, aug_trans_log)
            if delta_reset is None:
                perturb_batch = delta_batch.detach().requires_grad_()
            else:
                perturb_batch = load_delta(data_batch, soft_label, delta_batch, aug_policy, aug_trans_log)
            perturb_batch = perturb_batch.detach().requires_grad_()

            logits = model(data_batch + perturb_batch)
            if epoch_idx < warmup:
                target_this_batch = soft_label
            else:
                soft_prob = F.softmax(logits, dim = 1)
                for idx_in_batch, instance_idx in enumerate(idx_batch):
                    if instance_idx.__int__() in idx2target:
                        target_this_instance = idx2target[instance_idx.__int__()]
                    else:
                        target_this_instance = soft_label[idx_in_batch]
                    idx2target[instance_idx.__int__()] = (target_this_instance * rho + soft_prob[idx_in_batch] * (1. - rho)).detach()
                target_this_batch = [idx2target[instance_idx.__int__()].unsqueeze(0) for instance_idx in idx_batch]
                target_this_batch = torch.cat(target_this_batch, dim = 0)
                target_this_batch = (soft_label * beta + target_this_batch * (1. - beta)).detach()
            ce_loss = - (F.log_softmax(logits, dim = -1) * soft_label).sum(dim = 1).mean()
            grad = torch.autograd.grad(ce_loss, perturb_batch)[0]
            perturb_batch = torch.clamp(perturb_batch + torch.sign(grad) * step_size, min = - threshold, max = threshold)
            perturb_batch = torch.clamp(data_batch + perturb_batch, min = 0., max = 1.) - data_batch
            if delta_reset is not None:
                delta_batch = update_delta(data_batch, soft_label, delta_batch, perturb_batch, aug_policy, aug_trans_log)
                if delta_reset is not None:
                    for idx_in_batch, instance_idx in enumerate(idx_batch):
                        idx2delta[instance_idx.__int__()] = delta_batch[idx_in_batch].detach()

            adv_logits = model(data_batch + perturb_batch)
            size_this_batch = adv_logits.size(0)
            if loss_param['name'].lower() in ['ce',]:
                if reweight is None or epoch_idx < warmup_rw:
                    weight = torch.ones_like(label_batch).float()
                elif reweight.lower() in ['prob',]:
                    weight = F.softmax(adv_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['comp_prob',]:
                    weight = 1. - F.softmax(adv_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['max_prob',]:
                    weight = F.softmax(adv_logits, dim = -1).max(dim = 1)[0].detach()
                elif reweight.lower() in ['comp_max_prob',]:
                    weight = 1. - F.softmax(adv_logits, dim = -1).max(dim = 1)[0].detach()
                elif reweight.lower() in ['nat_prob',]:
                    nat_logits = model(data_batch)
                    weight = F.softmax(nat_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['comp_nat_prob',]:
                    nat_logits = model(data_batch)
                    weight = 1. - F.softmax(nat_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                else:
                    raise ValueError('Unrecognized reweight mode: %s' % reweight)
                loss = - (F.log_softmax(adv_logits, dim = -1) * target_this_batch).sum(dim = 1)
                loss = (loss * weight).sum() / weight.sum()
            elif loss_param['name'].lower() in ['trades',]:
                coefficient = loss_param['lambda'].__float__()
                nat_logits = model(data_batch)
                if reweight is None or epoch_idx < warmup_rw:
                    weight = torch.ones_like(label_batch).float()
                elif reweight.lower() in ['prob',]:
                    weight = F.softmax(adv_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['comp_prob',]:
                    weight = 1. - F.softmax(adv_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['nat_prob',]:
                    weight = F.softmax(nat_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['comp_nat_prob',]:
                    weight = 1. - F.softmax(nat_logits, dim = -1)[np.arange(size_this_batch), label_batch].detach()
                elif reweight.lower() in ['max_prob',]:
                    weight = F.softmax(adv_logits, dim = -1).max(dim = 1)[0].detach()
                elif reweight.lower() in ['comp_max_prob',]:
                    weight = 1. - F.softmax(adv_logits, dim = -1).max(dim = 1)[0].detach()
                elif reweight.lower() in ['nat_max_prob',]:
                    weight = F.softmax(nat_logits, dim = -1).max(dim = 1)[0].detach()
                elif reweight.lower() in ['comp_nat_max_prob',]:
                    weight = 1. - F.softmax(nat_logits, dim = -1).max(dim = 1)[0].detach()
                else:
                    raise ValueError('Unrecognized reweight mode: %s' % reweight)
                criterion_kl = nn.KLDivLoss(reduce = False)
                loss1 = - (F.log_softmax(nat_logits, dim = -1) * target_this_batch).sum(dim = 1)
                loss2 = criterion_kl(F.log_softmax(adv_logits, dim = -1), F.softmax(nat_logits, dim = -1)).sum(dim = 1)
                loss = loss1 + coefficient * loss2
                loss = (loss * weight).sum() / weight.sum()
            else:
                raise ValueError('Unrecognized Loss')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(adv_logits, label_batch)
            _, prediction_this_batch = logits.max(dim = 1)
            accuracy_bit_this_batch = (prediction_this_batch == label_batch).int().data.cpu().numpy()
            entropy_this_batch = - (F.log_softmax(adv_logits, dim = 1) * F.softmax(adv_logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
            loss_this_batch = - F.log_softmax(adv_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
            for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, accuracy_bit_this_batch, entropy_this_batch, loss_this_batch):
                acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
                entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()
                loss_per_instance_this_epoch[instance_idx.__int__()] = loss_this_instance.__float__()
            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

            if train_test == True:
                if test_attacker != None:
                    adv_data_batch, adv_label_batch = test_attacker.attack(model, data_batch, label_batch, criterion, False)
                else:
                    adv_data_batch, adv_label_batch = data_batch, label_batch
                logits_under_attack = model(adv_data_batch)
                loss_under_attack = criterion(logits_under_attack, adv_label_batch)
                acc_under_attack = accuracy(logits_under_attack.data, adv_label_batch)
                attack_loss_calculator.update(loss_under_attack.item(), data_batch.size(0))
                attack_acc_calculator.update(acc_under_attack.item(), data_batch.size(0))
                sys.stdout.write('Train - Instance Idx: %d - %.4f (%.4f) / %.2f%% (%.2f%%)\r' % (
                    idx, loss_calculator.average, attack_loss_calculator.average, acc_calculator.average * 100., attack_acc_calculator.average * 100.))
            else:
                sys.stdout.write('Train - Instance Idx: %d - %.4f / %.2f%%\r' % (idx, loss_calculator.average, acc_calculator.average * 100.))

        # Training summary
        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Train loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch
        tosave['train_acc_per_instance'][epoch_idx] = acc_per_instance_this_epoch
        tosave['train_entropy_per_instance'][epoch_idx] = entropy_per_instance_this_epoch
        tosave['train_loss_per_instance'][epoch_idx] = loss_per_instance_this_epoch

        if train_test == True:
            loss_under_attack_this_epoch = attack_loss_calculator.average
            acc_under_attack_this_epoch = attack_acc_calculator.average
            print('Train loss / acc after epoch under attack %d: %.4f / %.2f%%' % (epoch_idx, loss_under_attack_this_epoch, acc_under_attack_this_epoch * 100.))
            tosave['train_loss_under_attack'][epoch_idx] = loss_under_attack_this_epoch
            tosave['train_acc_under_attack'][epoch_idx] = acc_under_attack_this_epoch

        # Validation phease
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                    torch.save(model.module.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
                else:
                    torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                    torch.save(model.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))

                best_valid_acc = acc_this_epoch
                if len(best_valid_epoch) >= 1:
                    for best_valid_epoch_pt in best_valid_epoch[:-1]:
                        if best_valid_epoch_pt not in epoch_ckpts:
                            os.remove(os.path.join(out_folder, '%s_%d.ckpt' % (model_name, best_valid_epoch_pt)))
                best_valid_epoch = best_valid_epoch[-1:] + [epoch_idx + 1,]

        # Test phase
        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = test_loader,
            is_train = False, epoch_idx = epoch_idx, label = 'test', device = device, lr_func = None, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
            else:
                torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))
    else:
        torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave
