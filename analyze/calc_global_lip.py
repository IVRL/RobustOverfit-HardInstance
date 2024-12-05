import os
import sys
sys.path.insert(0, './')

import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from util.model_parser import parse_model
from util.param_parser import IntListParser
from util.device_parser import config_visible_gpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', action = IntListParser, default = [100, 3, 32, 32],
        help = 'The input size, default is [100, 3, 32, 32].')
    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset, default is "cifar10".')
    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The model type, default is "resnet".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to load. default is None.')

    parser.add_argument('--iter_num', type = int, default = 50,
        help = 'The iteration number of the power method, default is 50.')
    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file, default is None.')

    parser.add_argument('--loop', type = int, default = 20,
        help = 'The number of loops to calculate the Lipschitz constant, default is 20.')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default is None.')

    args = parser.parse_args()

    # Config GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse IO
    out_dir = os.path.dirname(args.out_file)
    if out_dir != '' and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # model
    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = None,)
    model = model.cuda() if use_gpu else model
    if args.model2load is not None:
        ckpt2load = torch.load(args.model2load)
        model.load_state_dict(ckpt2load)

    # Input
    data_batch = torch.randn(*args.input_size)
    data_batch = data_batch.cuda() if use_gpu else data_batch

    configs = {kwargs: value for kwargs, value in args._get_kwargs()}

    tosave = {'setup_config': configs, 'lip': None,
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    model.eval()
    with torch.no_grad():
        lip_list = [model[1].calc_lip(data_batch, iter_num = args.iter_num) for _ in range(args.loop)]
    lip_list_toprint = ['%1.3e' % lip.data.cpu().numpy().__float__() for lip in lip_list]
    print('The upper bound of the global Lipschitz constant:')
    print(lip_list_toprint)
    tosave['lip'] = [lip.data.cpu().numpy().__float__() for lip in lip_list]

    json.dump(tosave, open(args.out_file, 'w'))

