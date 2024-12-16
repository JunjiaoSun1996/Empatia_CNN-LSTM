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
@Time : Created in 2:51 PM 2024/02/20   
@FileName: Create_feature_maps_wesad.py                           
@Software: PyCharm


Introduction of this File:
We create the 2D feature maps here.
"""

import numpy as np
import random
from random import randint

from WESAD.Abnormal_extraction_weasd import data_extraction_wesad, data_prepare_wesad
from WESAD.Data_normalization_WESAD import Normalize_data_wesad

log_file = '../WEMAC/log_data_normalization_functions_KNN_AD.log'
data_file = 'json_files'


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


def get_normalization_wesad(data_path, log_path):
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label = data_extraction_wesad(json_dic=data_path)
    ## Using the normalization functions which calculated above to normalize data
    train_Nor_feature_array = Normalize_data_wesad(Normalization_log_file=log_path,
                                             feature_array=train_feature_array,
                                             If0to1=True)
    test_Nor_feature_array = Normalize_data_wesad(Normalization_log_file=log_path,
                                            feature_array=test_feature_array,
                                            If0to1=True)

    # Stack all the train and test. Include labels and P numbers.
    train_array_all = np.vstack((train_Nor_feature_array, train_label_array, train_P_label))
    test_array_all = np.vstack((test_Nor_feature_array, test_label_array, test_P_label))

    ## change nan to 0
    train_array_all[np.isnan(train_array_all)] = 0
    train_array_all[np.isinf(train_array_all)] = 0
    test_array_all[np.isnan(test_array_all)] = 0
    test_array_all[np.isinf(test_array_all)] = 0
    return train_array_all, test_array_all

def generate_feature_maps_wesad(features, strategy):
    all_feature_map, all_label = np.zeros(1), np.zeros(1)
    if strategy == 'AllFromOne':
        all_feature_map, all_label = AllFromOne_wesad(features)
    elif strategy == 'HalfAndHalf':
        all_feature_map, all_label = HalfAndHalf_wesad(features)
    elif strategy == 'HalfAndRandom':
        all_feature_map, all_label = HalfAndRandom_wesad(features)
    all_feature_map[np.isnan(all_feature_map)] = 0
    all_feature_map[np.isinf(all_feature_map)] = 0
    return all_feature_map, all_label


def AllFromOne_wesad(features):
    ## Strategy 1: All feature from 1 participant. And resize it for making the same H and W
    features_number = features.shape[0] - 2
    features = features.transpose()
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    L_set = sorted(set(features[:, features_number]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for L_num in L_set:
            Now_L_location = np.where(Now_P_data[:, features_number] == L_num)
            Now_L_data = Now_P_data[Now_L_location[0], :]
            # Define the parameters of feature map
            map_length = 60
            map_overlap = 15
            start_pose = 0
            while start_pose < Now_L_data.shape[0] - map_length:
                Now_feature_map = Now_L_data[start_pose:start_pose + map_length, :]
                feature_map_resized = np.resize(Now_feature_map[:, :features_number], (features_number, features_number))
                all_feature_map = np.dstack((all_feature_map, feature_map_resized))
                Now_label = Now_feature_map[0][features_number]
                all_label.append(int(Now_label))
                start_pose += map_overlap
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


def HalfAndHalf_wesad(features):
    ## Stratgy 2: Half from 1 participant and half from other random participant but same label
    features_number = features.shape[0] - 2
    features = features.transpose()
    feature_map = np.zeros(1)
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    L_set = sorted(set(features[:, features_number]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for L_num in L_set:
            Now_L_location = np.where(Now_P_data[:, features_number] == L_num)
            Now_L_data = Now_P_data[Now_L_location[0], :]
            Now_label = Now_L_data[0][features_number]
            all_L_data_location = np.where(features[:, features_number] == Now_label)
            all_L_data = features[all_L_data_location[0], :]
            random_P = random.choice(P_set)
            random_L_location = np.where(all_L_data[:, features_number + 1] == random_P)
            random_L_data = all_L_data[random_L_location[0], :]
            # Define the parameters of feature map
            map_length = 60
            map_overlap = 15
            start_pose = 0
            while start_pose < Now_L_data.shape[0] - map_length:
                Now_feature_map = Now_L_data[start_pose:start_pose + map_length, :]
                # randomly choose a piece of data from random_L_data
                random_index = randint(0, random_L_data.shape[0] - (features_number - map_length) - 5)
                random_feature_map = random_L_data[random_index:random_index + (features_number - map_length), :]
                inter_feature_map = np.vstack((Now_feature_map, random_feature_map))
                inter_feature_map = inter_feature_map[:, :features_number]
                all_feature_map = np.dstack((all_feature_map, inter_feature_map))
                all_label.append(int(Now_label))
                start_pose += map_overlap
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


def HalfAndRandom_wesad(features):
    ## Stratgy 3: Half from 1 video and half integrated by random Participants in same video
    features_number = features.shape[0] - 2
    features = features.transpose()
    feature_map = np.zeros(1)
    all_feature_map = np.zeros((features_number, features_number))
    all_label = []
    P_set = sorted(set(features[:, features_number + 1]))
    L_set = sorted(set(features[:, features_number]))
    for P_num in P_set:
        Now_P_location = np.where(features[:, features_number + 1] == P_num)
        Now_P_data = features[Now_P_location[0], :]
        for L_num in L_set:
            Now_L_location = np.where(Now_P_data[:, features_number] == L_num)
            Now_L_data = Now_P_data[Now_L_location[0], :]
            Now_label = Now_L_data[0][features_number]
            all_L_data_location = np.where(features[:, features_number] == Now_label)
            all_L_data = features[all_L_data_location[0], :]
            # Define the parameters of feature map
            map_length = 60
            map_overlap = 15
            start_pose = 0
            while start_pose < Now_L_data.shape[0] - map_length:
                Now_feature_map = Now_L_data[start_pose:start_pose + map_length, :]
                while Now_feature_map.shape[0] < features_number:
                    frame_index = randint(0, all_L_data.shape[0] - 1)
                    random_frame = all_L_data[frame_index, :]
                    Now_feature_map = np.vstack((Now_feature_map, random_frame))
                Now_feature_map = Now_feature_map[:, :features_number]
                all_feature_map = np.dstack((all_feature_map, Now_feature_map))
                all_label.append(int(Now_label))
                start_pose += map_overlap
    all_feature_map = all_feature_map[:, :, 1:]
    return all_feature_map, all_label


if __name__ == '__main__':
    train_array_all, test_array_all = get_normalization_wesad(data_file, log_file)
    test_feature_map, test_label = generate_feature_maps_wesad(test_array_all, strategy='AllFromOne')
    # test_feature_map, test_label = generate_feature_maps_wesad(test_array_all, strategy='HalfAndHalf')
    # test_feature_map, test_label = generate_feature_maps_wesad(test_array_all, strategy='HalfAndRandom')
    #
    train_feature_map, train_label = generate_feature_maps_wesad(train_array_all, strategy='AllFromOne')
    # train_feature_map, train_label = generate_feature_maps_wesad(train_array_all, strategy='HalfAndHalf')
    # train_feature_map, train_label = generate_feature_maps_wesad(train_array_all, strategy='HalfAndRandom')

    # Normalize the data to 0-255
    # test_feature_map = Normalize0to255(test_feature_map)
    # train_feature_map = Normalize0to255(train_feature_map)
    print(1)
