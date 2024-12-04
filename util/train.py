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

from .evaluation import *
from .utility import TRADES_Criterion_V2 as TRADES_Criterion
from .utility import SemiAda_Criterion
from .model_parser import get_features
from .optim_parser import set_optim_params

def epoch_pass(model, criterion, attacker, optimizer, loader, is_train, epoch_idx, label, device, lr_func, tosave):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    if is_train == True:
        model.train()
    else:
        model.eval()

    acc_per_instance_this_epoch = {}
    loss_per_instance_this_epoch = {}
    entropy_per_instance_this_epoch = {}
    for idx, (data_batch, label_batch, idx_batch) in enumerate(loader, 0):

        if is_train == True and lr_func is not None:
            epoch_batch_idx = epoch_idx
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learing rate = %1.2e' % lr_this_batch)

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch
        idx_batch = idx_batch.int().data.cpu().numpy()

        if attacker != None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion)
        else:
            adv_data_batch, adv_label_batch = data_batch, label_batch

        logits = model(adv_data_batch)
        loss = criterion(logits, adv_label_batch)
        acc = accuracy(logits.data, adv_label_batch)
        _, prediction_this_batch = logits.max(dim = 1)

        accuracy_bit_this_batch = (prediction_this_batch == adv_label_batch).int().data.cpu().numpy()
        entropy_this_batch = - (F.log_softmax(logits, dim = 1) * F.softmax(logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
        loss_this_batch = - F.log_softmax(logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
        for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, accuracy_bit_this_batch, entropy_this_batch, loss_this_batch):
            acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
            entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()
            loss_per_instance_this_epoch[instance_idx.__int__()] = loss_this_instance.__float__()

        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_calculator.update(loss.item(), data_batch.size(0))
        acc_calculator.update(acc.item(), data_batch.size(0))

        sys.stdout.write('%s - Instance Idx: %d - %.2f%%\r' % (label, idx, acc_calculator.average * 100.))

    loss_this_epoch = loss_calculator.average
    acc_this_epoch = acc_calculator.average
    print('%s loss / acc after epoch %d: %.4f / %.2f%%' % (label, epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
    tosave['%s_loss' % label][epoch_idx] = loss_this_epoch
    tosave['%s_acc' % label][epoch_idx] = acc_this_epoch
    tosave['%s_acc_per_instance' % label][epoch_idx] = acc_per_instance_this_epoch
    tosave['%s_entropy_per_instance' % label][epoch_idx] = entropy_per_instance_this_epoch
    tosave['%s_loss_per_instance' % label][epoch_idx] = loss_per_instance_this_epoch

    return model, tosave, loss_this_epoch, acc_this_epoch

def feature_analyze(model, criterion, attacker, loader, epoch_idx, label, device, idx2group, feature_mode, tosave):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    model.eval()
    feature_by_group = {}
    for idx, (data_batch, label_batch, idx_batch) in enumerate(loader, 0):

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch
        idx_batch = idx_batch.int().data.cpu().numpy()

        if attacker != None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion)
        else:
            adv_data_batch, adv_label_batch = data_batch, label_batch

        feature_this_batch = get_features(model = model, x = adv_data_batch, mode = feature_mode)
        feature_this_batch = feature_this_batch.data.cpu().numpy()

        for instance_idx, feature_this_instance in zip(idx_batch, feature_this_batch):
            group_this_instance = idx2group[int(instance_idx)]
            if group_this_instance not in feature_by_group:
                feature_by_group[group_this_instance] = [feature_this_instance.reshape(-1),]
            else:
                feature_by_group[group_this_instance].append(feature_this_instance.reshape(-1))

    tosave['%s_feature_mean' % label][epoch_idx] = {}
    tosave['%s_feature_median' % label][epoch_idx] = {}
    for group_idx in feature_by_group:
        feature_this_group = np.array(feature_by_group[group_idx])
        mean_this_group = np.mean(np.abs(feature_this_group), axis = 0)
        median_this_group = np.median(np.abs(feature_this_group), axis = 0)
        tosave['%s_feature_mean' % label][epoch_idx][group_idx] = [float(value) for value in mean_this_group]
        tosave['%s_feature_median' % label][epoch_idx][group_idx] = [float(value) for value in median_this_group]

    return model, tosave

def train(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, optimizer,
    lr_func, eps_func, out_folder, model_name, device, criterion, tosave, idx2group = None, feature_mode = None, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    for epoch_idx in range(epoch_num):

        # Training phase
        # Update the adversarial budget
        if eps_func is not None:
            threshold_this_epoch = eps_func(epoch_idx)
            train_attacker.adjust_threshold(threshold = threshold_this_epoch)

        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = train_attacker, optimizer = optimizer, loader = train_loader,
            is_train = True, epoch_idx = epoch_idx, label = 'train', device = device, lr_func = lr_func, tosave = tosave)

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = train_loader,
                epoch_idx = epoch_idx, label = 'train', device = device, idx2group = idx2group['train'], feature_mode = feature_mode, tosave = tosave)

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
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

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = test_loader,
                epoch_idx = epoch_idx, label = 'test', device = device, idx2group = idx2group['test'], feature_mode = feature_mode, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def train_instancewise(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, default_threshold, gamma, min_eps, max_eps, beta,
    epoch_num, warmup_epoch, epoch_ckpts, optimizer, lr_func, eps_func, mixed_loss, out_folder, model_name, device, criterion, tosave, idx2group = None, feature_mode = None, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    idx2threshold = {}

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    for epoch_idx in range(epoch_num):

        # Training phase
        # Update the adversarial budget
        if eps_func is not None:
            threshold_this_epoch = eps_func(epoch_idx)
            train_attacker.adjust_threshold(threshold = threshold_this_epoch)

        if epoch_idx < warmup_epoch:
            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = train_attacker, optimizer = optimizer, loader = train_loader,
                is_train = True, epoch_idx = epoch_idx, label = 'train', device = device, lr_func = lr_func, tosave = tosave)
        else:
            model.train()

            acc_calculator = AverageCalculator()
            loss_calculator = AverageCalculator()

            acc_per_instance_this_epoch = {}
            loss_per_instance_this_epoch = {}
            entropy_per_instance_this_epoch = {}
            for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

                epoch_batch_idx = epoch_idx
                lr_this_batch = lr_func(epoch_batch_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_batch
                if idx == 0:
                    tosave['lr_per_epoch'] = lr_this_batch
                    print('Learing rate = %1.2e' % lr_this_batch)

                data_batch = data_batch.cuda(device) if use_gpu else data_batch
                label_batch = label_batch.cuda(device) if use_gpu else label_batch
                idx_batch = idx_batch.int().data.cpu().numpy()

                # Construct threshold list
                threshold_list = [idx2threshold[int(idx)] if int(idx) in idx2threshold else default_threshold + min_eps for idx in idx_batch]
                threshold_inc_list = [threshold + gamma for threshold in threshold_list]

                train_attacker.adjust_threshold(threshold = threshold_inc_list)
                strong_adv_data_batch, _ = train_attacker.attack(model, data_batch, label_batch, criterion)
                _, prediction_under_strong_attack = model(strong_adv_data_batch).max(dim = 1)
                accuracy_bits_under_strong_attack = (prediction_under_strong_attack == label_batch).int().data.cpu().numpy()

                train_attacker.adjust_threshold(threshold = threshold_list)
                normal_adv_data_batch, _ = train_attacker.attack(model, data_batch, label_batch, criterion)
                _, prediction_under_normal_attack = model(normal_adv_data_batch).max(dim = 1)
                accuracy_bits_under_normal_attack = (prediction_under_normal_attack == label_batch).int().data.cpu().numpy()

                threshold_used_this_batch = []
                for instance_idx, accuracy_under_strong_attack, accuracy_under_normal_attack in zip(idx_batch, accuracy_bits_under_strong_attack, accuracy_bits_under_normal_attack):
                    threshold_last_iter = idx2threshold[int(instance_idx)] if int(instance_idx) in idx2threshold else default_threshold
                    if accuracy_under_strong_attack == 1:
                        idx2threshold[int(instance_idx)] = (1 - beta) * threshold_last_iter + beta * min(max_eps, threshold_last_iter + gamma)
                        threshold_used_this_batch.append(min(max_eps - min_eps, threshold_last_iter + gamma - min_eps))
                    elif accuracy_under_normal_attack == 1:
                        idx2threshold[int(instance_idx)] = threshold_last_iter
                        threshold_used_this_batch.append(threshold_last_iter - min_eps)
                    else:
                        idx2threshold[int(instance_idx)] = (1 - beta) * threshold_last_iter + beta * max(min_eps, threshold_last_iter - gamma)
                        threshold_used_this_batch.append(max(0, threshold_last_iter - gamma - min_eps))

                # Update parameter
                train_attacker.adjust_threshold(threshold = threshold_used_this_batch)
                adv_data_batch, _ = train_attacker.attack(model, data_batch, label_batch, criterion)

                adv_out = model(adv_data_batch)
                adv_accuracy_bits = (adv_out.max(dim = 1)[1] == label_batch).float()
                clean_out = model(data_batch)
                clean_accuracy_bits = (clean_out.max(dim = 1)[1] == label_batch).float()

                if mixed_loss == True:
                    out = adv_out * clean_accuracy_bits.view(-1, 1) + clean_out * (1. - clean_accuracy_bits.view(-1, 1))
                    loss = criterion(out, label_batch)
                else:
                    loss = criterion(adv_out, label_batch)
                acc = accuracy(adv_out.data, label_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                adv_target, _ = test_attacker.attack(model, data_batch, label_batch, criterion)
                logits_target = model(adv_target)
                accuracy_bits_target = (logits_target.max(dim = 1)[1] == label_batch).float()
                loss_target = criterion(logits_target, label_batch)
                acc_target = accuracy(logits_target.data, label_batch)

                accuracy_bit_this_batch = accuracy_bits_target.int().data.cpu().numpy()
                entropy_this_batch = - (F.log_softmax(logits_target, dim = 1) * F.softmax(logits_target, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
                loss_this_batch = - F.log_softmax(logits_target, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
                for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, accuracy_bit_this_batch, entropy_this_batch, loss_this_batch):
                    acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
                    entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()
                    loss_per_instance_this_epoch[instance_idx.__int__()] = loss_this_instance.__float__()

                loss_calculator.update(loss_target.item(), data_batch.size(0))
                acc_calculator.update(acc_target.item(), data_batch.size(0))

                sys.stdout.write('Train - Instance Idx: %d - %.2f%%\r' % (idx, acc_calculator.average * 100.))

            loss_this_epoch = loss_calculator.average
            acc_this_epoch = acc_calculator.average
            print('train loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
            tosave['train_loss'][epoch_idx] = loss_this_epoch
            tosave['train_acc'][epoch_idx] = acc_this_epoch
            tosave['train_idx2threshold'][epoch_idx] = deepcopy(idx2threshold)
            tosave['train_acc_per_instance'][epoch_idx] = acc_per_instance_this_epoch
            tosave['train_entropy_per_instance'][epoch_idx] = entropy_per_instance_this_epoch
            tosave['train_loss_per_instance'][epoch_idx] = loss_per_instance_this_epoch

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = train_loader,
                epoch_idx = epoch_idx, label = 'train', device = device, idx2group = idx2group['train'], feature_mode = feature_mode, tosave = tosave)

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
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

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = test_loader,
                epoch_idx = epoch_idx, label = 'test', device = device, idx2group = idx2group['test'], feature_mode = feature_mode, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def train_trades(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, gamma,
    optimizer, lr_func, eps_func, out_folder, model_name, device, criterion, tosave, idx2group = None, feature_mode = None, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    for epoch_idx in range(epoch_num):

        # Training phase
        # Update the adversarial budget
        if eps_func is not None:
            threshold_this_epoch = eps_func(epoch_idx)
            train_attacker.adjust_threshold(threshold = threshold_this_epoch)

        model.train()

        acc_calculator = AverageCalculator()
        loss_calculator = AverageCalculator()

        adv_acc_per_instance_this_epoch = {}
        adv_loss_per_instance_this_epoch = {}
        adv_entropy_per_instance_this_epoch = {}

        trades_acc_per_instance_this_epoch = {}
        trades_loss_per_instance_this_epoch = {}
        trades_entropy_per_instance_this_epoch = {}

        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

            epoch_batch_idx = epoch_idx
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learning rate = %1.2e' % lr_this_batch)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            idx_batch = idx_batch.int().data.cpu().numpy()

            clean_logits = model(data_batch)
            criterion_this_batch = TRADES_Criterion(fx = clean_logits)

            if train_attacker != None:
                trades_adv_data_batch, adv_label_batch = train_attacker.attack(model, data_batch, label_batch, criterion_this_batch)
                adv_data_batch, adv_label_batch = train_attacker.attack(model, data_batch, label_batch, criterion)
            else:
                trades_adv_data_batch, adv_label_batch = data_batch, label_batch
                adv_data_batch, adv_label_batch = data_batch, label_batch

            trades_logits = model(trades_adv_data_batch)
            adv_logits = model(adv_data_batch)

            loss2optimize = criterion(clean_logits, label_batch) + criterion_this_batch(trades_logits, label_batch) / gamma
            adv_acc = accuracy(adv_logits.data, adv_label_batch)

            optimizer.zero_grad()
            loss2optimize.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save
            _, trades_prediction_this_batch = trades_logits.max(dim = 1)
            _, adv_prediction_this_batch = adv_logits.max(dim = 1)

            trades_accuracy_bit_this_batch = (trades_prediction_this_batch == label_batch).int().data.cpu().numpy()
            adv_accuracy_bit_this_batch = (adv_prediction_this_batch == label_batch).int().data.cpu().numpy()
            trades_entropy_this_batch = - (F.log_softmax(trades_logits, dim = 1) * F.softmax(trades_logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
            adv_entropy_this_batch = - (F.log_softmax(adv_logits, dim = 1) * F.softmax(adv_logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
            trades_loss_this_batch = - F.log_softmax(trades_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
            adv_loss_this_batch = - F.log_softmax(adv_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()

            for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, adv_accuracy_bit_this_batch, adv_entropy_this_batch, adv_loss_this_batch):
                adv_acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
                adv_entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()
                adv_loss_per_instance_this_epoch[instance_idx.__int__()] = loss_this_instance.__float__()
            for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, trades_accuracy_bit_this_batch, trades_entropy_this_batch, trades_loss_this_batch):
                trades_acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
                trades_entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()
                trades_loss_per_instance_this_epoch[instance_idx.__int__()] = loss_this_instance.__float__()

            loss_calculator.update(loss2optimize.item(), data_batch.size(0))
            acc_calculator.update(adv_acc.item(), data_batch.size(0))

            sys.stdout.write('Training - Instance Idx: %d - %.2f%%\r' % (idx, acc_calculator.average * 100.))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Training loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch
        tosave['train_adv_acc_per_instance'][epoch_idx] = adv_acc_per_instance_this_epoch
        tosave['train_trades_acc_per_instance'][epoch_idx] = trades_acc_per_instance_this_epoch
        tosave['train_adv_entropy_per_instance'][epoch_idx] = adv_entropy_per_instance_this_epoch
        tosave['train_trades_entropy_per_instance'][epoch_idx] = trades_entropy_per_instance_this_epoch
        tosave['train_adv_loss_per_instance'][epoch_idx] = adv_loss_per_instance_this_epoch
        tosave['train_trades_loss_per_instance'][epoch_idx] = trades_loss_per_instance_this_epoch

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = train_loader,
                epoch_idx = epoch_idx, label = 'train', device = device, idx2group = idx2group['train'], feature_mode = feature_mode, tosave = tosave)

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
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

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = test_loader,
                epoch_idx = epoch_idx, label = 'test', device = device, idx2group = idx2group['test'], feature_mode = feature_mode, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def train_movetarget(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, epoch_warmup,
    alpha, gamma, optimizer, lr_func, eps_func, out_folder, model_name, device, criterion, tosave, idx2group = None, feature_mode = None, use_weight = True, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    train_idx2target = {}

    if use_weight == False:
        print('The weighting for different instances is disabled!')

    for epoch_idx in range(epoch_num):

        # Training phase
        if eps_func is not None:
            threshold_this_epoch = eps_func(epoch_idx)
            train_attacker.adjust_threshold(threshold = threshold_this_epoch)

        model.train()

        acc_ori_label_per_instance_this_epoch = {}
        acc_ada_label_per_instance_this_epoch = {}
        loss_ori_label_per_instance_this_epoch = {}
        loss_ada_label_per_instance_this_epoch = {}
        weight_per_instance_this_epoch = {}

        loss_calculator = AverageCalculator()
        acc_calculator = AverageCalculator()

        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

            epoch_batch_idx = epoch_idx
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learning rate = %1.2e' % lr_this_batch)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            idx_batch = idx_batch.int().data.cpu().numpy()

            clean_logits = model(data_batch)
            clean_probs = F.softmax(clean_logits, dim = -1)
            criterion_this_batch = TRADES_Criterion(fx = clean_logits)

            # Construct soft labels
            if epoch_idx == 0: # Initialize
                one_hot_label = torch.zeros_like(clean_logits)
                one_hot_label.scatter_(1, label_batch.view(-1, 1), 1)
                for idx_this_batch, instance_idx in enumerate(idx_batch):
                    train_idx2target[instance_idx] = one_hot_label[idx_this_batch].data
            elif epoch_idx >= epoch_warmup: # Update
                for idx_this_batch, instance_idx in enumerate(idx_batch):
                    train_idx2target[instance_idx] = alpha * train_idx2target[instance_idx].data + (1. - alpha) * clean_probs[idx_this_batch].data
            target = [train_idx2target[instance_idx].view(1, -1) for instance_idx in idx_batch]
            target = torch.cat(target, dim = 0)
            weight, target_prediction = torch.max(target, dim = 1)
            if use_weight == False:     # Cancel the weight if it is disabled
                weight = torch.ones_like(weight)

            if train_attacker != None:
                trades_data_batch, adv_label_batch = train_attacker.attack(model, data_batch, label_batch, criterion_this_batch)
                adv_data_batch, adv_label_batch = train_attacker.attack(model, data_batch, label_batch, criterion)
            else:
                trades_data_batch, adv_label_batch = data_batch.data, label_batch.data
                adv_data_batch, adv_label_batch = data_batch.data, label_batch.data

            unweighted_clean_loss = - (F.log_softmax(clean_logits) * target).sum(dim = 1)
            weighted_clean_loss = (unweighted_clean_loss * weight).sum() / weight.sum()

            trades_logits = model(trades_data_batch)
            adv_logits = model(adv_data_batch)
            loss2optimize = weighted_clean_loss + criterion_this_batch(trades_logits, label_batch) / gamma

            optimizer.zero_grad()
            loss2optimize.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save
            _, clean_prediction = clean_logits.max(dim = 1)
            adv_acc = accuracy(adv_logits.data, label_batch)

            acc_ori_label_this_batch = (clean_prediction == label_batch).float().data.cpu().numpy()
            acc_ada_label_this_batch = (clean_prediction == target_prediction).float().data.cpu().numpy()
            loss_ori_label_this_batch = - F.log_softmax(clean_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
            loss_ada_label_this_batch = unweighted_clean_loss.data.cpu().numpy()
            weight_this_batch = weight.data.cpu().numpy()

            for instance_idx, acc_ori_label, acc_ada_label, loss_ori_label, loss_ada_label, weight_this_instance in zip(
                idx_batch, acc_ori_label_this_batch, acc_ada_label_this_batch, loss_ori_label_this_batch, loss_ada_label_this_batch, weight_this_batch):
                acc_ori_label_per_instance_this_epoch[instance_idx.__int__()] = acc_ori_label.__int__()
                acc_ada_label_per_instance_this_epoch[instance_idx.__int__()] = acc_ada_label.__int__()
                loss_ori_label_per_instance_this_epoch[instance_idx.__int__()] = loss_ori_label.__float__()
                loss_ada_label_per_instance_this_epoch[instance_idx.__int__()] = loss_ada_label.__float__()
                weight_per_instance_this_epoch[instance_idx.__int__()] = weight_this_instance.__float__()

            loss_calculator.update(loss2optimize.item(), data_batch.size(0))
            acc_calculator.update(adv_acc.item(), data_batch.size(0))

            sys.stdout.write('Training - Instance Idx: %d - %.2f%%\r' % (idx, acc_calculator.average * 100.))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Training loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch
        tosave['train_acc_ori_label_per_instance'][epoch_idx] = acc_ori_label_per_instance_this_epoch
        tosave['train_acc_ada_label_per_instance'][epoch_idx] = acc_ada_label_per_instance_this_epoch
        tosave['train_loss_ori_label_per_instance'][epoch_idx] = loss_ori_label_per_instance_this_epoch
        tosave['train_loss_ada_label_per_instance'][epoch_idx] = loss_ada_label_per_instance_this_epoch
        tosave['train_weight_per_instance'][epoch_idx] = weight_per_instance_this_epoch

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = train_loader,
                epoch_idx = epoch_idx, label = 'train', device = device, idx2group = idx2group['train'], feature_mode = feature_mode, tosave = tosave)

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
                train_idx2target_tosave = {key: train_idx2target[key].data.cpu().numpy() for key in train_idx2target}

                torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
                pickle.dump(train_idx2target_tosave, open(os.path.join(out_folder, '%s_%d_idx2target.pkl' % (model_name, epoch_idx + 1)), 'wb'))
                torch.save(model.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
                pickle.dump(train_idx2target_tosave, open(os.path.join(out_folder, '%s_bestvalid_idx2target.pkl' % model_name), 'wb'))

                best_valid_acc = acc_this_epoch
                if len(best_valid_epoch) >= 1:
                    for best_valid_epoch_pt in best_valid_epoch[:-1]:
                        if best_valid_epoch_pt not in epoch_ckpts:
                            os.remove(os.path.join(out_folder, '%s_%d.ckpt' % (model_name, best_valid_epoch_pt)))
                            os.remove(os.path.join(out_folder, '%s_%d_idx2target.pkl' % (model_name, best_valid_epoch_pt)))
                best_valid_epoch = best_valid_epoch[-1:] + [epoch_idx + 1,]

        # Test phase
        model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = test_loader,
            is_train = False, epoch_idx = epoch_idx, label = 'test', device = device, lr_func = None, tosave = tosave)

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = test_loader,
                epoch_idx = epoch_idx, label = 'test', device = device, idx2group = idx2group['test'], feature_mode = feature_mode, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            train_idx2target_tosave = {key: train_idx2target[key].data.cpu().numpy() for key in train_idx2target}
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
            pickle.dump(train_idx2target_tosave, open(os.path.join(out_folder, '%s_%d_idx2target.pkl' % (model_name, epoch_idx + 1)), 'wb'))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    train_idx2target_tosave = {key: train_idx2target[key].data.cpu().numpy() for key in train_idx2target}
    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def train_mart(model, train_loader, valid_loader, test_loader, train_attacker, test_attacker, epoch_num, epoch_ckpts, gamma, optimizer, lr_func, eps_func,
    out_folder, model_name, device, criterion, tosave, use_bce = False, use_ada_kl = False, idx2group = None, feature_mode = None, **tricks):

    best_valid_acc = 0.
    best_valid_epoch = []

    valid_acc_last_epoch = 0.

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    for epoch_idx in range(epoch_num):

        # Training phase
        if eps_func is not None:
            threshold_this_epoch = eps_func(epoch_idx)
            train_attacker.adjust_threshold(threshold = threshold_this_epoch)

        model.train()

        acc_calculator = AverageCalculator()
        loss_calculator = AverageCalculator()

        acc_per_instance_this_epoch = {}
        ce_loss_per_instance_this_epoch = {}
        kl_loss_per_instance_this_epoch = {}
        entropy_per_instance_this_epoch = {}

        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

            epoch_batch_idx = epoch_idx
            lr_this_batch = lr_func(epoch_batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_batch
            if idx == 0:
                tosave['lr_per_epoch'] = lr_this_batch
                print('Learning rate = %1.2e' % lr_this_batch)

            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            idx_batch = idx_batch.int().data.cpu().numpy()

            clean_logits = model(data_batch)

            if train_attacker != None:
                adv_data_batch, adv_label_batch = train_attacker.attack(model, data_batch, label_batch, criterion)
            else:
                adv_data_batch, adv_label_batch = data_batch, label_batch

            adv_logits = model(adv_data_batch)
            _, prediction_this_batch = adv_logits.max(dim = 1)

            ce_per_instance_this_batch = - F.log_softmax(adv_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            kl_per_instance_this_batch = (F.softmax(clean_logits, dim = 1) * (F.log_softmax(clean_logits, dim = 1) - F.log_softmax(adv_logits, dim = 1))).sum(dim = 1)

            if use_bce == True:
                value, indices = F.softmax(adv_logits, dim = 1).topk(2, dim = 1)
                acc_bits = (indices[:, 0] == label_batch).float()
                first_false = value[:, 0] * (1. - acc_bits) + value[:, 1] * acc_bits
                bce = - torch.log(1. - first_false + 1e-8)
            else:
                bce = 0

            if use_ada_kl:
                ada_kl = kl_per_instance_this_batch * (1. - F.softmax(clean_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1))
            else:
                ada_kl = kl_per_instance_this_batch

            loss = (ce_per_instance_this_batch + bce + ada_kl * gamma).mean()
            acc = accuracy(adv_logits.data, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save
            ce_per_instance_this_batch = ce_per_instance_this_batch.data.cpu().numpy()
            kl_per_instance_this_batch = kl_per_instance_this_batch.data.cpu().numpy()
            accuracy_bit_this_batch = (prediction_this_batch == label_batch).int().data.cpu().numpy()
            entropy_this_batch = - (F.log_softmax(adv_logits, dim = 1) * F.softmax(adv_logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()

            for instance_idx, accuracy_bit, entropy_this_instance, ce_loss_this_instance, kl_loss_this_instance in zip(
                idx_batch, accuracy_bit_this_batch, entropy_this_batch, ce_per_instance_this_batch, kl_per_instance_this_batch):

                acc_per_instance_this_epoch[instance_idx.__int__()] = accuracy_bit.__int__()
                ce_loss_per_instance_this_epoch[instance_idx.__int__()] = ce_loss_this_instance.__float__()
                kl_loss_per_instance_this_epoch[instance_idx.__int__()] = kl_loss_this_instance.__float__()
                entropy_per_instance_this_epoch[instance_idx.__int__()] = entropy_this_instance.__float__()

            loss_calculator.update(ce_per_instance_this_batch.mean().item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

            sys.stdout.write('Training - Instance Idx: %d - %.2f%%\r' % (idx, acc_calculator.average * 100.))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Training loss / acc after epoch %d: %.4f / %.2f%%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch
        tosave['train_acc_per_instance'][epoch_idx] = acc_per_instance_this_epoch
        tosave['train_entropy_per_instance'][epoch_idx] = entropy_per_instance_this_epoch
        tosave['train_ce_loss_per_instance'][epoch_idx] = ce_loss_per_instance_this_epoch
        tosave['train_kl_loss_per_instance'][epoch_idx] = kl_loss_per_instance_this_epoch

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = train_loader,
                epoch_idx = epoch_idx, label = 'train', device = device, idx2group = idx2group['train'], feature_mode = feature_mode, tosave = tosave)

        # Validation phase
        if valid_loader is not None:

            model, tosave, loss_this_epoch, acc_this_epoch = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None, loader = valid_loader,
                is_train = False, epoch_idx = epoch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)

            if acc_this_epoch < valid_acc_last_epoch:
                print('Validation accuracy decreases!')
            else:
                print('Validation accuracy increases!')
            valid_acc_last_epoch = acc_this_epoch

            if acc_this_epoch > best_valid_acc:
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

        if idx2group is not None:
            model, tosave = feature_analyze(model = model, criterion = criterion, attacker = test_attacker, loader = test_loader,
                epoch_idx = epoch_idx, label = 'test', device = device, idx2group = idx2group['test'], feature_mode = feature_mode, tosave = tosave)

        if (epoch_idx + 1) in epoch_ckpts and (epoch_idx + 1) not in best_valid_epoch:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))
        json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def loss_in_tune(model, data_batch, label_batch, attacker, criterion, device, tune_mode = 'base', tune_params = None):

    if tune_mode is None or tune_mode in ['base',]:
        ada_weight = False if 'ada_weight' not in tune_params or int(tune_params['ada_weight']) == 0 else True
        max_prob_as_weight = True if 'max_prob_as_weight' not in tune_params or int(tune_params['max_prob_as_weight']) != 0 else False
        diff_weight = True if 'diff_weight' not in tune_params or int(tune_params['diff_weight']) != 0 else False

        if attacker != None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion)
        else:
            adv_data_batch, adv_label_batch = data_batch, label_batch

        logits = model(adv_data_batch)
        if ada_weight == True:
            clean_logits = model(data_batch)
            ce_by_instance = - F.log_softmax(logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if max_prob_as_weight == True:
                weight_by_instance = F.softmax(clean_logits, dim = 1).max(dim = 1)[0]
            else:
                weight_by_instance = F.softmax(clean_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
            if diff_weight == True:
                loss = (ce_by_instance * weight_by_instance).sum() / weight_by_instance.sum() 
            else:
                loss = (ce_by_instance * weight_by_instance.data).sum() / weight_by_instance.data.sum()
        else:
            loss = criterion(logits, adv_label_batch)
    elif tune_mode in ['self_ada',]:
        gamma = float(tune_params['gamma']) if 'gamma' in tune_params else 0.5
        ada_attack = False if 'ada_attack' not in tune_params or int(tune_params['ada_attack']) == 0 else True

        clean_logits = model(data_batch)
        ada_criterion = SemiAda_Criterion(fx = clean_logits, gamma = gamma)
        if attacker is not None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion if ada_attack is False else ada_criterion)
        else:
            adv_data_batch, adv_label_batch = data_batch, label_batch
        logits = model(adv_data_batch)
        loss = model(adv_data_batch)
    elif tune_mode in ['kl',]:
        gamma = float(tune_params['gamma']) if 'gamma' in tune_params else 6
        ada_weight = True if 'ada_weight' not in tune_params or int(tune_params['ada_weight']) != 0 else False
        max_prob_as_weight = True if 'max_prob_as_weight' not in tune_params or int(tune_params['max_prob_as_weight']) != 0 else False
        diff_weight = True if 'diff_weight' not in tune_params or int(tune_params['diff_weight']) != 0 else False

        clean_logits = model(data_batch)
        if attacker is not None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion)
        else:
            adv_data_batch, adv_label_batch = data_batch, label_batch

        logits = model(adv_data_batch)

        ce_by_instance = - F.log_softmax(logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1)
        kl_by_instance = (F.softmax(clean_logits, dim = 1) * (F.log_softmax(clean_logits, dim = 1) - F.log_softmax(logits, dim = 1))).sum(dim = 1)
        loss_by_instance = ce_by_instance + gamma * kl_by_instance

        if max_prob_as_weight == True:
            weight_by_instance = F.softmax(clean_logits, dim = 1).max(dim = 1)[0] if ada_weight is True else torch.ones_like(loss_by_instance)
        else:
            weight_by_instance = F.softmax(clean_logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1) if ada_weight is True else torch.ones_like(loss_by_instance)
        if diff_weight == True:
            loss = (loss_by_instance * weight_by_instance).sum() / weight_by_instance.sum()
        else:
            loss = (loss_by_instance * weight_by_instance.data).sum() / weight_by_instance.data.sum()
    else:
        raise ValueError('Unrecognized tuning mode: %s' % tune_mode)

    return logits, loss

def finetune(model, train_loader, valid_loader, test_loader, valid_freq, train_attacker, test_attacker, epoch_num, epoch_ckpts, optimizer,
    lr_func, out_folder, model_name, device, criterion, tosave, tune_mode = 'base', tune_params = None, **tricks):

    best_valid_acc = 0.
    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    acc_calculator = AverageCalculator()
    loss_calculator = AverageCalculator()

    global_batch_idx = 0
    global_batch_idx_tested = []

    for epoch_idx in range(epoch_num):

        # Training phase
        for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):

            model.train()
            if lr_func is not None:
                lr_this_batch = lr_func(epoch_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_batch
                if idx == 0:
                    print('Learning rate = %1.2e' % lr_this_batch)

            global_batch_idx += 1
            data_batch = data_batch.cuda(device) if use_gpu else data_batch
            label_batch = label_batch.cuda(device) if use_gpu else label_batch
            idx_batch = idx_batch.int().data.cpu().numpy()

            logits, loss = loss_in_tune(model, data_batch, label_batch, train_attacker, criterion, device, tune_mode, tune_params)

            acc = accuracy(logits.data, label_batch)
            _, prediction_this_batch = logits.max(dim = 1)

            accuracy_bit_this_batch = (prediction_this_batch == label_batch).int().data.cpu().numpy()
            entropy_this_batch = - (F.log_softmax(logits, dim = 1) * F.softmax(logits, dim = 1)).sum(dim = 1).float().data.cpu().numpy()
            loss_this_batch = - F.log_softmax(logits, dim = 1).gather(dim = 1, index = label_batch.view(-1, 1)).view(-1).float().data.cpu().numpy()
            for instance_idx, accuracy_bit, entropy_this_instance, loss_this_instance in zip(idx_batch, accuracy_bit_this_batch, entropy_this_batch, loss_this_batch):
                instance_idx = instance_idx.__int__()
                if instance_idx not in tosave['train_idx2acc']:
                    tosave['train_idx2acc'][instance_idx] = [accuracy_bit.__int__(),]
                else:
                    tosave['train_idx2acc'][instance_idx].append(accuracy_bit.__int__())
                if instance_idx not in tosave['train_idx2entropy']:
                    tosave['train_idx2entropy'][instance_idx] = [entropy_this_instance.__float__(),]
                else:
                    tosave['train_idx2entropy'][instance_idx].append(entropy_this_instance.__float__())
                if instance_idx not in tosave['train_idx2loss']:
                    tosave['train_idx2loss'][instance_idx] = [loss_this_instance.__float__(),]
                else:
                    tosave['train_idx2loss'][instance_idx].append(loss_this_instance.__float__())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_calculator.update(loss.item(), data_batch.size(0))
            acc_calculator.update(acc.item(), data_batch.size(0))

            sys.stdout.write('Train - Instance Idx: %d - %.2f%%\r' % (idx, acc_calculator.average * 100.))

            # Do validation and testing
            if global_batch_idx % valid_freq == 0:
                print('')
                global_batch_idx_tested.append(global_batch_idx)

                model.eval()
                if valid_loader is not None:
                    model, tosave, valid_loss_this_point, valid_acc_this_point = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None,
                        loader = valid_loader, is_train = False, epoch_idx = global_batch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)
                    if valid_acc_this_point > best_valid_acc:
                        torch.save(model.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
                        best_valid_acc = valid_acc_this_point

                model, tosave, test_loss_this_point, test_acc_this_point = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None,
                    loader = test_loader, is_train = False, epoch_idx = global_batch_idx, label = 'test', device = device, lr_func = None, tosave = tosave)

                json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))

        loss_this_epoch = loss_calculator.average
        acc_this_epoch = acc_calculator.average
        print('Training loss / acc after epoch %d: %.4f / %.2f %%' % (epoch_idx, loss_this_epoch, acc_this_epoch * 100.))
        tosave['train_loss'][epoch_idx] = loss_this_epoch
        tosave['train_acc'][epoch_idx] = acc_this_epoch

        if epoch_idx + 1 in epoch_ckpts:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_%d.ckpt' % (model_name, epoch_idx + 1)))

    # Do validation and test in the end of training
    if global_batch_idx not in global_batch_idx_tested:

        model.eval()
        model, tosave, valid_loss_this_point, valid_acc_this_point = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None,
            loader = valid_loader, is_train = False, epoch_idx = global_batch_idx, label = 'valid', device = device, lr_func = None, tosave = tosave)
        if valid_acc_this_point > best_valid_acc:
            torch.save(model.state_dict(), os.path.join(out_folder, '%s_bestvalid.ckpt' % model_name))
            best_valid_acc = valid_acc_this_point

        model, tosave, test_loss_this_point, test_acc_this_point = epoch_pass(model = model, criterion = criterion, attacker = test_attacker, optimizer = None,
            loader = test_loader, is_train = False, epoch_idx = global_batch_idx, label = 'test', device = device, lr_func = None, tosave = tosave)

    json.dump(tosave, open(os.path.join(out_folder, '%s.json' % model_name), 'w'))
    torch.save(model.state_dict(), os.path.join(out_folder, '%s.ckpt' % model_name))

    return model, tosave

def attack(model, loader, attacker, device, criterion, **tricks):

    use_gpu = device != torch.device('cpu') and torch.cuda.is_available()

    clean_acc_calculator = AverageCalculator()
    clean_loss_calculator = AverageCalculator()
    adv_acc_calculator = AverageCalculator()
    adv_loss_calculator = AverageCalculator()

    clean_acc_bits = []
    adv_acc_bits = []

    model.eval()
    for idx, (data_batch, label_batch, idx_batch) in enumerate(loader, 0):

        data_batch = data_batch.cuda(device) if use_gpu else data_batch
        label_batch = label_batch.cuda(device) if use_gpu else label_batch

        clean_logits = model(data_batch)
        clean_loss = criterion(clean_logits, label_batch)
        clean_acc = accuracy(clean_logits.data, label_batch)

        _, clean_prediction = clean_logits.max(dim = 1)
        clean_acc_bits_this_batch = (clean_prediction == label_batch).data.float().cpu().numpy()
        clean_acc_bits = clean_acc_bits + list(clean_acc_bits_this_batch)

        clean_loss_calculator.update(clean_loss.item(), data_batch.size(0))
        clean_acc_calculator.update(clean_acc.item(), data_batch.size(0))

        if attacker is not None:
            adv_data_batch, adv_label_batch = attacker.attack(model, data_batch, label_batch, criterion)

            adv_logits = model(adv_data_batch)
            adv_loss = criterion(adv_logits, adv_label_batch)
            adv_acc = accuracy(adv_logits.data, adv_label_batch)

            _, adv_prediction = adv_logits.max(dim = 1)
            adv_acc_bits_this_batch = (adv_prediction == adv_label_batch).data.float().cpu().numpy()
            adv_acc_bits = adv_acc_bits + list(adv_acc_bits_this_batch)

            adv_loss_calculator.update(adv_loss.item(), adv_data_batch.size(0))
            adv_acc_calculator.update(adv_acc.item(), adv_data_batch.size(0))

            sys.stdout.write('Instance Idx: %d, Clean Acc = %.2f%%, Robust Acc = %.2f%%\r' % (idx, clean_acc_calculator.average * 100., adv_acc_calculator.average * 100.))
        else:
            sys.stdout.write('Instance Idx: %d, Clean Acc = %.2f%%\r' % (idx, clean_acc_calculator.average * 100.))

    print('Clean loss / acc: %.4f / %.2f%%' % (clean_loss_calculator.average, clean_acc_calculator.average * 100.))
    if attacker != None:
        print('Robust loss / acc: %.4f / %.2f%%' % (adv_loss_calculator.average, adv_acc_calculator.average * 100.))

    return clean_acc_bits, adv_acc_bits
