"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 4:38 PM 2024/02/19   
@FileName: Create_feature_maps.py                           
@Software: PyCharm


Introduction of this File:
This file include the functions about how to create 2D feature maps.
We designed two ways for data augmentation include: half and half, half and random.
"""
import sys

import numpy as np
from random import choice
from random import randint
sys.path.append("/home/junjiao/PycharmProjects/Empatia_Git/WEMAC")

from Data_extraction import data_extraction, Abnormal_detection
from Data_normalization import Normalize_data

log_file = 'log_data_normalization_functions_KNN_AD.log'
data_file = 'Pack_all_data.json'


def Normalize0to255(array):
    """
    Normalize the array
    """
    for index in range(array.shape[2]):
        # Remember !!! The data in same column is the same feature.
        for index_f in range(array.shape[1]):
            mx = np.nanmax(array[:, :, index][:, index_f])
            mn = np.nanmin(array[:, :, index][:, index_f])
            # Suppress/hide the warning
            np.seterr(invalid='ignore')
            array[:, :, index][:, index_f] = np.round((array[:, :, index][:, index_f] - mn) / (mx - mn) * 255)
    array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array


def get_normalization(data_path, log_path):
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, \
        train_V_label, test_V_label = data_extraction(data_path)
    train_filter_array = Abnormal_detection(train_feature_array)
    test_filter_array = Abnormal_detection(test_feature_array)
    ## Using the normalization functions which calculated above to normalize data
    train_Nor_feature_array = Normalize_data(Normalization_log_file=log_path,
                                             feature_array=train_filter_array,
                                             If0to1=True)
    test_Nor_feature_array = Normalize_data(Normalization_log_file=log_path,
                                            feature_array=test_filter_array,
                                            If0to1=True)
    # Transfer the P_labels to array
    train_P_label = np.array([int(i.replace('P', '')) for i in train_P_label])
    test_P_label = np.array([int(i.replace('P', '')) for i in test_P_label])
    # Transfer the V_labels to array
    train_V_label = np.array([float(i) for i in train_V_label])
    test_V_label = np.array([float(i) for i in test_V_label])
    # Stack all the train and test. Include labels and P numbers.
    train_array_all = np.vstack((train_Nor_feature_array, train_label_array, train_P_label, train_V_label))
    test_array_all = np.vstack((test_Nor_feature_array, test_label_array, test_P_label, test_V_label))

    ## change nan to 0
    train_array_all[np.isnan(train_array_all)] = 0
    train_array_all[np.isinf(train_array_all)] = 0
    test_array_all[np.isnan(test_array_all)] = 0
    test_array_all[np.isinf(test_array_all)] = 0
    return train_array_all, test_array_all


def generate_feature_maps(features, strategy):
    all_feature_map, all_label = np.zeros(1), np.zeros(1)
    if strategy == 'AllFromOne':
        all_feature_map, all_label = AllFromOne(features)
    elif strategy == 'HalfAndHalf':
        all_feature_map, all_label = HalfAndHalf(features)
    elif strategy == 'HalfAndRandom':
        all_feature_map, all_label = HalfAndRandom(features)
    all_feature_map[np.isnan(all_feature_map)] = 0
    all_feature_map[np.isinf(all_feature_map)] = 0
    return all_feature_map, all_label


def AllFromOne(features):
    ## Stratgy 1: All feature from 1 video. And resize it for making the same H and W
    features_number = features.shape[0] - 3
    features = features.transpose()
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    V_set = sorted(set(features[:, features_number + 2]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for V_num in V_set:
            Now_V_location = np.where(Now_P_data[:, features_number + 2] == V_num)
            if Now_V_location[0] != []:
                Now_V_data = Now_P_data[Now_V_location[0], :]
                feature_map_resized = np.resize(Now_V_data[:, :features_number], (features_number, features_number))
                all_feature_map = np.dstack((all_feature_map, feature_map_resized))
                Now_label = Now_V_data[0][features_number]
                all_label.append(int(Now_label))
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


def HalfAndHalf(features):
    ## Stratgy 2: Half from 1 video and half from other random pariticipant but same video
    features_number = features.shape[0] - 3
    features = features.transpose()
    feature_map = np.zeros(1)
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    V_set = sorted(set(features[:, features_number + 2]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for V_num in V_set:
            Now_V_location = np.where(Now_P_data[:, features_number + 2] == V_num)
            Now_V_data = Now_P_data[Now_V_location[0], :]
            V_label = Now_V_data[0, features_number]
            # Pick all video data in same num and in same label
            all_V_data_location = np.where((features[:, features_number + 2] == V_num) &
                                           (features[:, features_number] == V_label))
            all_V_data = features[all_V_data_location[0], :]
            P_set_sameLabel = sorted(set(all_V_data[:, features_number + 1]))
            all_V_inter = []
            for index, num in enumerate(P_set_sameLabel):
                V_same = np.where(all_V_data[:, features_number + 1] == num)
                len_index = len(V_same[0])
                all_V_inter.append(len_index)
            # integrate two data.
            Random_inter = randint(0, len(all_V_inter) - 1)
            for index in range(len(Now_V_data)):
                if index == 0:
                    feature_map = Now_V_data[index]
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
                else:
                    feature_map = np.vstack((feature_map, Now_V_data[index]))
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
            feature_map_resized = np.resize(feature_map[:, :features_number], (features_number, features_number))
            all_feature_map = np.dstack((all_feature_map, feature_map_resized))
            all_label.append(V_label)
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


def HalfAndRandom(features):
    ## Stratgy 2: Half from 1 video and half integrated by random Participants in same video
    features_number = features.shape[0] - 3
    features = features.transpose()
    feature_map = np.zeros(1)
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    V_set = sorted(set(features[:, features_number + 2]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for V_num in V_set:
            Now_V_location = np.where(Now_P_data[:, features_number + 2] == V_num)
            Now_V_data = Now_P_data[Now_V_location[0], :]
            V_label = Now_V_data[0, features_number]
            # Pick all video data in same num and in same label
            all_V_data_location = np.where((features[:, features_number + 2] == V_num) &
                                           (features[:, features_number] == V_label))
            all_V_data = features[all_V_data_location[0], :]
            P_set_sameLabel = sorted(set(all_V_data[:, features_number + 1]))
            all_V_inter = []
            for index, num in enumerate(P_set_sameLabel):
                V_same = np.where(all_V_data[:, features_number + 1] == num)
                len_index = len(V_same[0])
                all_V_inter.append(len_index)
            # integrate two data.
            for index in range(len(Now_V_data)):
                if index == 0:
                    feature_map = Now_V_data[index]
                    Random_inter = randint(0, len(all_V_inter) - 1)
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
                else:
                    feature_map = np.vstack((feature_map, Now_V_data[index]))
                    Random_inter = randint(0, len(all_V_inter) - 1)
                    try:
                        feature_map = np.vstack((feature_map, all_V_data[index + sum(all_V_inter[:Random_inter])]))
                    except:
                        feature_map = np.vstack((feature_map, Now_V_data[index]))
            feature_map_resized = np.resize(feature_map[:, :features_number], (features_number, features_number))
            all_feature_map = np.dstack((all_feature_map, feature_map_resized))
            all_label.append(V_label)
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


if __name__ == '__main__':
    train_array_all, test_array_all = get_normalization(data_file, log_file)
    test_feature_map, test_label = generate_feature_maps(test_array_all, strategy='AllFromOne')
    # test_feature_map, test_label = generate_feature_maps(test_array_all, strategy='HalfAndHalf')
    # test_feature_map, test_label = generate_feature_maps(test_array_all, strategy='HalfAndRandom')
    #
    train_feature_map, train_label = generate_feature_maps(train_array_all, strategy='AllFromOne')
    # train_feature_map, train_label = generate_feature_maps(train_array_all, strategy='HalfAndHalf')
    # train_feature_map, train_label = generate_feature_maps(train_array_all, strategy='HalfAndRandom')

    # Normalize the data to 0-255
    # test_feature_map = Normalize0to255(test_feature_map)
    # train_feature_map = Normalize0to255(train_feature_map)
    print(1)
