import os
import sys
sys.path.insert(0, './')

import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt

import json
import argparse
import numpy as np

from util.color import get_color
from util.param_parser import ListParser, FloatListParser

def plot_curve(dict_data, linewidth, linestyle, color, alpha, label = None, x_range = None):

    if dict_data is None:
        return

    keys = [int(key) for key in dict_data.keys()]
    if x_range is not None:
        keys = list(filter(lambda x: True if x >= x_range[0] and x < x_range[1] else False, keys))
    keys = list(sorted(keys))
    values = [1. - dict_data[str(idx)] for idx in keys]

    if label is None:
        plt.plot(keys, values, alpha = alpha, linewidth = linewidth, linestyle = linestyle, color = color)
    else:
        plt.plot(keys, values, label = label, alpha = alpha, linewidth = linewidth, linestyle = linestyle, color = color)

def plot(json_files, labels, subsets, x_range):

    if labels is None:
        labels = [None,] * len(json_files)
    assert len(json_files) == len(labels), 'The length of json_files and labels must be the same'

    for idx, (json_file, label) in enumerate(zip(json_files, labels)):

        color = get_color(idx)
        data = json.load(open(json_file, 'r'))

        train_acc = data['train_acc']
        valid_acc = data['valid_acc']
        test_acc = data['test_acc']

        if '0' in subsets:
            plot_curve(dict_data = train_acc, linestyle = '--', linewidth = 2, color = color, alpha = 1., label = label, x_range = x_range)
        if '1' in subsets:
            plot_curve(dict_data = valid_acc, linestyle = ':', linewidth = 2, color = color, alpha = 1., label = label if '0' not in subsets else None, x_range = x_range)
        if '2' in subsets:
            plot_curve(dict_data = test_acc, linestyle = '-', linewidth = 2, color = color, alpha = 1., label = label if '0' not in subsets and '1' not in subsets else None, x_range = x_range)

    if len(list(filter(lambda x: x is not None, labels))) > 0:
        plt.legend(prop = {'size': 12})
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--json_files', action = ListParser, default = None,
        help = 'The list of json files.')
    parser.add_argument('--labels', action = ListParser, default = None,
        help = 'The label of json files.')

    parser.add_argument('--subsets', type = str, default = '012',
        help = 'The subsets to plot, default is "012".')

    parser.add_argument('--x_range', action = FloatListParser, default = None,
        help = 'The x-range to plot, default is None.')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file, default is None.')

    args = parser.parse_args()

    plot(json_files = args.json_files, labels = args.labels, subsets = args.subsets, x_range = args.x_range)

    out_folder = os.path.dirname(args.out_file)
    if out_folder != '' and os.path.exists(out_folder) == False:
        os.makedirs(out_folder)
    plt.savefig(args.out_file, dpi = 500, bbox = 'tight')

