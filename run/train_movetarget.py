import os
import sys
sys.path.insert(0, './')
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.attack import parse_attacker
from util.train import train_movetarget
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.data_parser import parse_data, parse_c_data, parse_subset, parse_groups
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The nomralization mode, default is None.')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default is 128.')
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'The number of epochs, default is 200.')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The checkpoint epoch, default is [].')
    parser.add_argument('--epoch_warmup', type = int, default = 0,
        help = 'The number of warmup epochs, default is 0.')

    parser.add_argument('--alpha', type = float, default = 0.9,
        help = 'The coefficient of moving target, default is 0.9.')
    parser.add_argument('--gamma', type = float, default = 1.,
        help = 'The coefficient applied to the second term of trades, default is 1.')

    parser.add_argument('--per_file', type = str, default = None,
        help = 'The per file.')
    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default is None.')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder.')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model.')

    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-1, 'momentum': 0.9, 'weight_decay': 5e-4},
        help = 'The optimizer, default is name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4.')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'jump', 'start_v': 1e-1, 'min_jump_pt': 100, 'jump_freq': 50, 'power': 0.1},
        help = 'The learning rate schedule, default is name=jump,min_jump_pt=100,jump_freq=50,start_v=0.1,power=0.1.')
    parser.add_argument('--eps_schedule', action = DictParser, default = None,
        help = 'The scheduler of the adversarial budget, default is None')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None, use name=h to obtain help messages.')
    parser.add_argument('--corrupt', action = DictParser, default = None,
        help = 'Whether or not to use the corrupted dataset, default is None.')

    parser.add_argument('--feature_groups', type = int, default = None,
        help = 'The number of feature groups, default is None.')
    parser.add_argument('--feature_mode', type = str, default = None,
        help = 'The feature mode, default is None.')
    parser.add_argument('--use_weight', action = BooleanParser, default = True,
        help = 'Weight or not to use the non-uniform weight, default is True.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default is None.')

    args = parser.parse_args()

    # Config the GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model and dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, valid_ratio = args.valid_ratio)
    if args.corrupt is not None:
        _, _, test_loader, _ = parse_c_data(name = args.dataset, batch_size = args.batch_size, **args.corrupt)

    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    if args.model2load is not None:
        ckpt2load = torch.load(args.model2load)
        model.load_state_dict(ckpt2load)

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None
    eps_func = continuous_seq(**args.eps_schedule) if args.eps_schedule != None else None

    # Parse the attacker
    train_attacker = None if args.attack == None else parse_attacker(**args.attack)
    test_attacker = None if args.attack == None or args.corrupt is not None else parse_attacker(**args.attack)

    # Parse idx2group
    if args.feature_groups is not None:
        idx2group = parse_groups(per_file = args.per_file, group_num = args.feature_groups)
    else:
        idx2group = None

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'train_acc': {},
        'train_acc_ori_label_per_instance': {}, 'train_acc_ada_label_per_instance': {}, 'train_loss_ori_label_per_instance': {}, 'train_loss_ada_label_per_instance': {}, 'train_weight_per_instance': {},
        'valid_loss': {}, 'valid_acc': {}, 'valid_acc_per_instance': {}, 'valid_entropy_per_instance': {}, 'valid_loss_per_instance': {},
        'test_loss': {}, 'test_acc': {}, 'test_acc_per_instance': {}, 'test_entropy_per_instance': {}, 'test_loss_per_instance': {},
        'total_batch_num': {}, 'lr_per_epoch': {}, 'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    if idx2group is not None:
        tosave.update({'train_feature_mean': {}, 'train_feature_median': {}, 'test_feature_mean': {}, 'test_feature_median': {}})

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    train_movetarget(model = model, train_loader = train_loader, valid_loader = valid_loader, test_loader = test_loader, train_attacker = train_attacker, test_attacker = test_attacker,
        epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, epoch_warmup = args.epoch_warmup, alpha = args.alpha, gamma = args.gamma, optimizer = optimizer, lr_func = lr_func, eps_func = eps_func,
        out_folder = args.out_folder, model_name = args.model_name, device = device, criterion = criterion, tosave = tosave, idx2group = idx2group, feature_mode = args.feature_mode, use_weight = args.use_weight)

