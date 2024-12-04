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
from util.train import finetune
from util.seq_parser import continuous_seq
from util.model_parser import parse_model
from util.optim_parser import parse_optim
from util.device_parser import config_visible_gpu
from util.data_parser import parse_plus_data, parse_subset, parse_groups
from util.param_parser import DictParser, IntListParser, FloatListParser, BooleanParser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The nomralization mode, default is None.')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')
    parser.add_argument('--valid_freq', type = int, default = 400,
        help = 'The frequency to conduct the validation phase, default is 400.')
    parser.add_argument('--plus_prop', type = float, default = 0.25,
        help = 'The proportion of the additional data, default is 0.25')

    parser.add_argument('--per_file', type = str, default = None,
        help = 'The per file to load, default is None.')
    parser.add_argument('--subset', action = DictParser, default = None,
        help = 'The type of the subset, default is None, format is mode=XXX,num=DDD')

    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'The batch size, default is 128.')
    parser.add_argument('--epoch_num', type = int, default = 1,
        help = 'The number of the additional epochs, default is 1.')
    parser.add_argument('--epoch_ckpts', action = IntListParser, default = [],
        help = 'The number of ckpt to save, default is []')

    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default is None.')

    parser.add_argument('--out_folder', type = str, default = None,
        help = 'The output folder.')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'The name of the model.')

    parser.add_argument('--optim', action = DictParser, default = {'name': 'sgd', 'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 5e-4},
        help = 'The optimizer, default is name=sgd,lr=1e-2,momentum=0.9,weight_decay=5e-4.')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'const', 'start_v': 1e-2,},
        help = 'The learning rate schedule, default is name=const,start_v=0.01.')

    parser.add_argument('--attack', action = DictParser, default = None,
        help = 'Play adversarial attack or not, default = None, use name=h to obtain help messages.')

    parser.add_argument('--tune_mode', type = str, default = 'base', choices = ['base', 'self_ada', 'kl'],
        help = 'The mode of fine tuning, default is "base", supported = ["base", "self_ada"]')
    parser.add_argument('--tune_params', action = DictParser, default = {},
        help = 'The additional parameters of tuning. default is {}.')

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
    train_subset = None if args.subset is None else parse_subset(per_file = args.per_file, subset = args.subset)
    train_loader, valid_loader, test_loader, classes = parse_plus_data(name = args.dataset, batch_size = args.batch_size, plus_prop = args.plus_prop, valid_ratio = args.valid_ratio, train_subset = train_subset)

    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize)
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    assert args.model2load is not None, 'A pretrained model should be identified.'
    ckpt2load = torch.load(args.model2load)
    model.load_state_dict(ckpt2load)

    # Parse the optimizer
    optimizer = parse_optim(policy = args.optim, params = model.parameters())
    lr_func = continuous_seq(**args.lr_schedule) if args.lr_schedule != None else None

    # Parse the attacker
    train_attacker = None if args.attack == None else parse_attacker(**args.attack)
    test_attacker = None if args.attack == None else parse_attacker(**args.attack, mode = 'test')

    # Prepare the item to save
    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'train_loss': {}, 'train_acc': {}, 'train_idx2acc': {}, 'train_idx2entropy': {}, 'train_idx2loss': {},
        'valid_loss': {}, 'valid_acc': {}, 'valid_acc_per_instance': {}, 'valid_entropy_per_instance': {}, 'valid_loss_per_instance': {},
        'test_loss': {}, 'test_acc': {}, 'test_acc_per_instance': {}, 'test_entropy_per_instance': {}, 'test_loss_per_instance': {},
        'lr_per_epoch': {}, 'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    for param in list(sorted(tosave['setup_config'].keys())):
        print('%s\t=>%s' % (param, tosave['setup_config'][param]))

    finetune(model = model, train_loader = train_loader, valid_loader = valid_loader, test_loader = test_loader, valid_freq = args.valid_freq,
        train_attacker = train_attacker, test_attacker = test_attacker, epoch_num = args.epoch_num, epoch_ckpts = args.epoch_ckpts, optimizer = optimizer, lr_func = lr_func,
        out_folder = args.out_folder, model_name = args.model_name, device = device, criterion = criterion, tosave = tosave, tune_mode = args.tune_mode, tune_params = args.tune_params)

