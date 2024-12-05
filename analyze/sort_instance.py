import os
import sys
sys.path.insert(0, './')
import json
import argparse

import numpy as np
from datetime import datetime

from util.color import get_color
from util.data_parser import parse_data
from util.param_parser import IntListParser, FloatListParser, BooleanParser

def gen_sort(acc_report, idx2label, reverse, min_epoch, mode = 'sum'):

    valid_epoch_num = 0

    # Generate idx2value
    idx2value = {}
    if mode.lower() in ['sum',]:
        for key in acc_report:
            if int(key) < min_epoch:
                continue
            valid_epoch_num += 1
            for idx in acc_report[key]:
                if int(idx) not in idx2value:
                    idx2value[int(idx)] = 0
                idx2value[int(idx)] += acc_report[key][idx]
    elif mode.lower() in ['last',]:
        key_list = list(map(int, acc_report.keys()))
        key_list = list(sorted(key_list))
        key = str(key_list[-1])
        for idx in acc_report[key]:
            if int(idx) not in idx2value:
                idx2value[int(idx)] = 0
            idx2value[int(idx)] += acc_report[key][idx]
        valid_epoch_num += 1
    else:
        raise ValueError('Unrecognized mode: %s' % mode)

    instance_num = len(idx2value)
    print('#Valid Epoch = %d, #Valid Instance = %d' % (valid_epoch_num, instance_num))

    # Generate value2per
    value2num = {}
    for idx in idx2value:
        if idx2value[idx] not in value2num:
            value2num[idx2value[idx]] = 0
        value2num[idx2value[idx]] += 1
    value_list = list(value2num.keys())
    value_list = list(sorted(value_list, reverse = reverse))

    value2per = {}
    accumul = 0
    for value in value_list:
        num = value2num[value]
        value2per[value] = [(accumul + num / 2) / instance_num, (accumul, accumul + num)]
        accumul += num
    assert accumul == instance_num

    # Organize 
    per_report = {}
    for idx in idx2value:
        label = idx2label[int(idx)]
        value = idx2value[int(idx)]
        average = value / valid_epoch_num
        per = value2per[value]
        if label not in per_report:
            per_report[label] = []
        per_report[label].append({'per': per, 'value': value, 'average': average, 'label': label, 'idx': int(idx)})
    for label in per_report:
        per_report[label] = list(sorted(per_report[label], key = lambda x: x['per']))

    return per_report

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',type = str, default = 'cifar10',
        help = 'The dataset to use, default is "cifar10".')
    parser.add_argument('--valid_ratio', type = float, default = None,
        help = 'The proportion of the validation set, default is None.')

    parser.add_argument('--min_epoch', type = int, default = 0,
        help = 'The minimum epoch index to consider, default is 0.')
    parser.add_argument('--mode', type = str, default = 'sum',
        help = 'The mode, default is "sum".')

    parser.add_argument('--json_file', type = str, default = 'The json file to load',
        help = 'The json file to load.')
    parser.add_argument('--metric', type = str, default = 'acc', choices = ['acc', 'loss', 'entropy'],
        help = 'The metric to sort easy or hard examples, default is "acc", ["acc", "loss", "entropy"] are supported.')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file.')

    args = parser.parse_args()

    data = json.load(open(args.json_file, 'r'))

    train_loader, valid_loader, test_loader, classes = parse_data(name = args.dataset, batch_size = 100, valid_ratio = args.valid_ratio)

    train_idx2label = {}
    valid_idx2label = {}
    test_idx2label = {}

    for idx, (data_batch, label_batch, idx_batch) in enumerate(train_loader, 0):
        label_batch = label_batch.data.cpu().numpy()
        idx_batch = idx_batch.cpu().numpy()
        for idx_this_instance, label_this_instance in zip(idx_batch, label_batch):
            train_idx2label[int(idx_this_instance)] = int(label_this_instance)

    for idx, (data_batch, label_batch, idx_batch) in enumerate(valid_loader, 0):
        label_batch = label_batch.data.cpu().numpy()
        idx_batch = idx_batch.cpu().numpy()
        for idx_this_instance, label_this_instance in zip(idx_batch, label_batch):
            valid_idx2label[int(idx_this_instance)] = int(label_this_instance)

    for idx, (data_batch, label_batch, idx_batch) in enumerate(test_loader, 0):
        label_batch = label_batch.data.cpu().numpy()
        idx_batch = idx_batch.cpu().numpy()
        for idx_this_instance, label_this_instance in zip(idx_batch, label_batch):
            test_idx2label[int(idx_this_instance)] = int(label_this_instance)

    trainset_key = 'train_%s_per_instance' % args.metric.lower()
    validset_key = 'valid_%s_per_instance' % args.metric.lower()
    testset_key = 'test_%s_per_instance' % args.metric.lower()

    reverse = False if args.metric.lower() in ['acc',] else True

    per_report_train = gen_sort(acc_report = data[trainset_key], idx2label = train_idx2label, reverse = reverse, min_epoch = args.min_epoch, mode = args.mode)
    per_report_valid = gen_sort(acc_report = data[validset_key], idx2label = valid_idx2label, reverse = reverse, min_epoch = args.min_epoch, mode = args.mode)
    per_report_test = gen_sort(acc_report = data[testset_key], idx2label = test_idx2label, reverse = reverse, min_epoch = args.min_epoch, mode = args.mode)

    configs = {kwargs: value for kwargs, value in args._get_kwargs()}

    tosave = {'setup_config': configs, 'train_per_report': per_report_train, 'valid_per_report': per_report_valid, 'test_per_report': per_report_test,
        'log': {'cmd': 'python ' + ' '.join(sys.argv), 'time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')}}

    # Parse IO
    out_folder = os.path.dirname(args.out_file)
    if out_folder != '' and os.path.exists(out_folder) is False:
        os.makedirs(out_folder)
    json.dump(tosave, open(args.out_file, 'w'))

