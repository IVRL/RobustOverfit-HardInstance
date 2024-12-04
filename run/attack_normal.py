import os
import sys
sys.path.insert(0, './')

import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from datetime import datetime

from util.train import attack
from util.attack import parse_attacker
from util.data_parser import parse_data
from util.model_parser import parse_model
from util.param_parser import DictParser
from util.device_parser import config_visible_gpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'cifar10',
        help = 'The dataset used, default = "cifar10".')
    parser.add_argument('--model_type', type = str, default = 'resnet',
        help = 'The type of the model, default is "resnet".')
    parser.add_argument('--normalize', type = str, default = None,
        help = 'The normalization mode, default is None.')
    parser.add_argument('--batch_size', type = int, default = 100,
        help = 'The batch size, default = 100.')

    parser.add_argument('--subset', type = str, default = 'test',
        help = 'Specify which set is used, default = "test".')
    parser.add_argument('--model2load', type = str, default = None,
        help = 'The model to be loaded, default = None.')

    parser.add_argument('--per_file', type = str, default = None,
        help = 'The profile file, default is None.')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file, default is None, meaning no file to save the results')

    parser.add_argument('--attack', action = DictParser, default = {'name': 'APGD', 'threshold': 8., 'order': -1, 'iter_num': 200, 'rho': 0.75},
        help = 'Play adversarial attack or not, default = None, default isX')

    parser.add_argument('--loop', type = int, default = 1,
        help = 'The number of PGD attack loops, default = 1')

    parser.add_argument('--gpu', type = str, default = None,
        help = 'Specify the GPU to use, default = None.')

    args = parser.parse_args()

    # Config the GPU
    config_visible_gpu(args.gpu)
    use_gpu = args.gpu != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # Parse model and dataset
    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = args.batch_size, shuffle = False)
    model = parse_model(dataset = args.dataset, model_type = args.model_type, normalize = args.normalize)

    loader = {'train': train_loader, 'test': test_loader}[args.subset]
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if use_gpu else model
    criterion = criterion.cuda() if use_gpu else criterion

    ckpt2load = torch.load(args.model2load)
    model.load_state_dict(ckpt2load)

    # Parse the attacker
    attacker = None if args.attack is None else parse_attacker(**args.attack)

    configs = {kwargs: value for kwargs, value in args._get_kwargs()}
    tosave = {'model_summary': str(model), 'setup_config': configs, 'clean_acc_per_run': [], 'adv_acc_per_run': [], 'final_adv_acc_bits_hard2easy': None,
        'final_adv_acc': None, 'final_adv_acc_bits': None, 'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    final_accuracy_bits = None
    for idx in range(args.loop):
        print('Scan -- %d' % idx)
        clean_accuracy_bits, adv_accuracy_bits = attack(model = model, loader = loader, attacker = attacker, device = device, criterion = criterion)
        if final_accuracy_bits is None:
            final_accuracy_bits = np.array(adv_accuracy_bits)
        else:
            final_accuracy_bits = final_accuracy_bits * np.array(adv_accuracy_bits)
        tosave['clean_acc_per_run'].append(np.mean(clean_accuracy_bits).__float__())
        tosave['adv_acc_per_run'].append(np.mean(adv_accuracy_bits).__float__())

    assert max(tosave['clean_acc_per_run']) - min(tosave['clean_acc_per_run']) < 1e-4, 'The clean accuracy should be the same for different runs'

    tosave['final_clean_acc'] = (max(tosave['clean_acc_per_run']) + min(tosave['clean_acc_per_run'])) / 2.
    tosave['final_adv_acc'] = np.mean(final_accuracy_bits).__float__()
    tosave['final_adv_acc_bits'] = [float(bits) for bits in final_accuracy_bits]

    print('>>> Final results')
    print('Clean Accuracy: %.2f%%' % (tosave['final_clean_acc'] * 100.))
    print('Robust Accuracy: %.2f%%' % (np.mean(final_accuracy_bits) * 100.))

    if args.per_file is not None:

        correct_num = [0 for _ in range(10)]
        total_num = [0 for _ in range(10)]

        per_data = json.load(open(args.per_file, 'r'))
        for label in per_data['test_per_report']:
            for item in per_data['test_per_report'][label]:
                index_this_item = item['idx']
                per_this_item = item['per'][0]
                group_this_item = min(max(per_this_item // 0.1, 0), 9).__int__()

                correct_num[group_this_item] += final_accuracy_bits[index_this_item]
                total_num[group_this_item] += 1

        correct_rate = [correct / total for correct, total in zip(correct_num, total_num)]
        correct_rate_print = ['%.4f' % rate for rate in correct_rate]
        tosave['correct_num_by_group'] = correct_num
        tosave['total_num_by_group'] = total_num
        tosave['correct_rate_by_group'] = correct_rate

        print('Correct Number: ', correct_num)
        print('Total Number: ', total_num)
        print('Correct Rate: ', correct_rate_print)

    # Parse IO
    out_folder = os.path.dirname(args.out_file)
    if out_folder != '' and os.path.exists(out_folder) is False:
        os.makedirs(out_folder)
    json.dump(tosave, open(args.out_file, 'w'))
