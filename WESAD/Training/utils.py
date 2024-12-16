'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import shutil
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from Feature_wise_Normalization.Create_feature_maps import generate_feature_maps, get_normalization
from skfeature.function.similarity_based import fisher_score

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def set_reconstruct(train_feature_map, test_feature_map, nowTrain_label, nowTest_label):
    train_feature_map_new = np.dstack((train_feature_map, test_feature_map[:, :, :int(test_feature_map.shape[2] / 3)]))
    nowTrain_label = np.hstack((nowTrain_label, nowTest_label[:int(nowTest_label.shape[0] / 3)]))
    return train_feature_map_new, test_feature_map, nowTrain_label, nowTest_label

_, term_width = shutil.get_terminal_size()
# _, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def Feature_selection_fisher_score_wesad(feature_array_all):
    feature_num = feature_array_all.shape[0] - 2
    feature_map = feature_array_all[:feature_num, :]
    feature_map = feature_map.transpose()
    label = feature_array_all[feature_num, :]
    rank = fisher_score.fisher_score(feature_map, label, mode='rank')
    # generate dic to save ranking
    rank_dic = {}
    for index, value in enumerate(rank):
        rank_dic[index] = value
    return rank_dic

def Feature_selection_fisher_score_empatia(feature_array_all):
    feature_num = feature_array_all.shape[0] - 3
    feature_map = feature_array_all[:feature_num, :]
    feature_map = feature_map.transpose()
    label = feature_array_all[feature_num, :]
    rank = fisher_score.fisher_score(feature_map, label, mode='rank')
    # generate dic to save ranking
    rank_dic = {}
    for index, value in enumerate(rank):
        rank_dic[index] = value
    return rank_dic

if __name__ == '__main__':
    log_file = '../Feature_wise_Normalization/log_data_normalization_functions_KNN_AD.log'
    data_file = '../Feature_wise_Normalization/Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file)
    all_array = np.hstack((train_array_all, test_array_all))

    dic = Feature_selection_fisher_score(feature_array_all=all_array)
