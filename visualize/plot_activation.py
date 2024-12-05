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
from util.param_parser import IntListParser, ListParser, FloatListParser

def plot(json_file, subset, group_list, label_list, metric, norm):

    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    data = json.load(open(json_file, 'r'))
    feature_mean = data['%s_feature_%s' % (subset, metric)]
    norm = norm if norm > 0 else np.inf

    epoch_list = list(feature_mean.keys())
    epoch_list = list(sorted(map(int, epoch_list)))

    # Plot Magnitude
    for idx, (group, label) in enumerate(zip(group_list, label_list)):
        color_this_group = get_color(idx)
        feature_trace_this_group = [np.linalg.norm(feature_mean[str(epoch_idx)][str(group)], ord = norm) for epoch_idx in epoch_list]
        ax_left.plot(epoch_list, feature_trace_this_group, color = color_this_group, label = label, linewidth = 3)

    # Plot Learn Curve
    train_err = [1. - data['train_acc'][str(epoch_idx)] for epoch_idx in epoch_list]
    test_err = [1. - data['test_acc'][str(epoch_idx)] for epoch_idx in epoch_list]
    ax_right.plot(epoch_list, train_err, color = 'k', alpha = 0.3, linewidth = 3, linestyle = '--')
    ax_right.plot(epoch_list, test_err, color = 'k', alpha = 0.3, linewidth = 3, linestyle = '-')

    ax_left.legend(prop = {'size': 12})
    ax_left.set_xlabel('Epoch')
    ax_left.set_ylabel('Feature Magnitude')
    ax_right.set_ylabel('Error')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--json_file', type = str, default = None,
        help = 'The json file.')
    parser.add_argument('--subset', type = str, default = 'train',
        help = 'The subset used, default is "train".')
    parser.add_argument('--group_list', action = IntListParser, default = None,
        help = 'The list of groups to plot.')
    parser.add_argument('--label_list', action = ListParser, default = None,
        help = 'The list of labels to plot.')
    parser.add_argument('--metric', type = str, default = 'mean', choices = ['mean', 'median'],
        help = 'The metric of aggregation, default is "mean", supported = ["mean", “median”]')
    parser.add_argument('--norm', type = int, default = 2,
        help = 'The norm used to calculate the magnitude, default is 2')

    parser.add_argument('--out_file', type = str, default = None,
        help = 'The output file, default is None.')

    args = parser.parse_args()

    plot(json_file = args.json_file, subset = args.subset, group_list = args.group_list, label_list = args.label_list, metric = args.metric, norm = args.norm)

    out_folder = os.path.dirname(args.out_file)
    if out_folder != '' and os.path.exists(out_folder) == False:
        os.makedirs(out_folder)
    plt.savefig(args.out_file, dpi = 500, bbox_inches = 'tight')

