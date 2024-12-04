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
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.data_parser import parse_data, parse_subset
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser, ListParser

from algorithm.adv_fast import adv_train_fast

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')
    parser.add_argument('--aug_policy', action = ListParser, default = ['crop', 'vflip'],
        help = 'The data augmentation, default is ["crop", "vflip"].')

    parser.add_argument('--per_file', type = str, default = None,
        help = 'The per file containing the instance difficulty information.')
    parser.add_argument('--train_subset', action = DictParser, default = None,
        help = 'The training subset used, default is None.')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default is 128.')
    parser.add_argument('--epoch_num', type = int, default = 200,
        help = 'The number of epochs, default is 200.')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The checkpoint epoch, default is [].')

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

    parser.add_argument('--loss_param', action = DictParser, default = {'name': 'ce'},
        help = 'The loss function, default is name=ce.')
    parser.add_argument('--reweight', type = str, default = None,
        help = 'The reweighting mode, default is None.')

    parser.add_argument('--threshold', type = float, default = None,
        help = 'The size of the adversarial budget, default is None.')
    parser.add_argument('--step_size', type = float, default = None,
        help = 'The step size of the attacker, default is None.')
    parser.add_argument('--delta_reset', type = int, default = None,
        help = 'The frequency of resetting delta, default is None, meaning no use of delta.')

    parser.add_argument('--rho', type = float, default = 0.,
        help = 'The coefficient to mix up the current prediction and the moving target, default is 0.')
    parser.add_argument('--beta', type = float, default = 0.,
        help = 'The coefficient to mix up the moving target and the one-hot ground truth, default is 0.')
    parser.add_argument('--warmup', type = int, default = 0,
        help = 'The warmup period, default is 0.')
    parser.add_argument('--warmup_rw', type = int, default = 0,
        help = 'The warmup period for reweighting, default is 0.')

    parser.add_argument('--test_attack', action = DictParser, default = None,
        help = 'The attacker used in the test time, default is None.')
    parser.add_argument('--train_test', action = BooleanParser, default = False,
        help = 'To track the training status under test attacker, default is False.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default is None.')

    args = parser.parse_args()

    # Config the GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    gpu_num = torch.cuda.device_count()
    print('GPU number = %d' % gpu_num)

    # Parse IO
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Parse model and dataset
    if args.train_subset is not None:
        subset_idx = parse_subset(per_file = args.per_file, subset = args.train_subset)
    else:
        subset_idx = None
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, valid_ratio = args.valid_ratio, augmentation = False, train_subset = subset_idx)
    output_dim = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200}[args.dataset]

    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = None, activation_func = 'relu')
    if args.model2load is not None:
        ckpt2load = torch.load(args.model2load, map_location = torch.device('cpu'))
        model.load_state_dict(ckpt2load)
    if gpu_num > 1:
        model = nn.DataParallel(model, device_ids = [gpu_idx for gpu_idx in range(gpu_num)])
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None

    # Parse the attacker
    threshold = args.threshold / 255. if args.threshold > 1. else args.threshold
    step_size = args.step_size / 255. if args.step_size > 1. else args.step_size
    test_attacker = None if args.test_attack == None else parse_attacker(**args.test_attack)

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'train_acc': {}, 'train_acc_per_instance': {}, 'train_entropy_per_instance': {}, 'train_loss_per_instance': {},
        'valid_loss': {}, 'valid_acc': {}, 'valid_acc_per_instance': {}, 'valid_entropy_per_instance': {}, 'valid_loss_per_instance': {},
        'test_loss': {}, 'test_acc': {}, 'test_acc_per_instance': {}, 'test_entropy_per_instance': {}, 'test_loss_per_instance': {}, 'train_loss_under_attack': {}, 'train_acc_under_attack': {},
        'total_batch_num': {}, 'lr_per_epoch': {}, 'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    model, tosave = adv_train_fast(model = model, output_dim = output_dim, train_loader = train_loader, valid_loader = valid_loader, test_loader = test_loader, aug_policy = args.aug_policy,
        threshold = threshold, step_size = step_size, loss_param = args.loss_param, reweight = args.reweight, rho = args.rho, beta = args.beta, warmup = args.warmup, warmup_rw = args.warmup_rw,
        train_test = args.train_test, delta_reset = args.delta_reset, test_attacker = test_attacker, epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, optimizer = optimizer,
        lr_func = lr_func, out_folder = args.out_folder, model_name = args.model_name, device = device, criterion = criterion, tosave = tosave)
