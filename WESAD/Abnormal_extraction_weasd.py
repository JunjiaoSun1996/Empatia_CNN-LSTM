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
@Time : Created in 2:48 PM 2024/02/20   
@FileName: Abnormal_extraction_weasd.py                           
@Software: PyCharm


Introduction of this File:
We put data extraction function in this file.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from WEMAC.Data_extraction import get_train_test


def data_extraction_wesad(json_dic='json_files'):
    feature_dic_all = {}
    train_feature_all_dic = {}
    train_label_all_dic = {}
    test_feature_all_dic = {}
    test_label_all_dic = {}
    # Load every feature from Weasd
    for (root, dirs, files) in os.walk(json_dic):
        for file in files:
            file_list = file.split('_')
            file_num = file_list[-1].replace('.json', '')
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                content = json.load(f)
            feature_dic_all[file_num] = content
    train_P_list, test_P_list = get_train_test(feature_dic_all)
    for participant in feature_dic_all.items():
        for episode in participant[1].values():
            for feature_name in episode.keys():
                # Judge if the feature from the participants from train or test
                if participant[0] in train_P_list:
                    if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                        if feature_name not in train_feature_all_dic.keys():
                            train_feature_all_dic[feature_name] = [episode[feature_name]]
                        else:
                            train_feature_all_dic[feature_name].append(episode[feature_name])
                    elif ('label' in feature_name):
                        if feature_name not in train_label_all_dic.keys():
                            train_label_all_dic[feature_name] = [episode[feature_name]]
                            # put the number of participant to label
                            train_label_all_dic['Participant'] = [participant[0]]
                        else:
                            train_label_all_dic[feature_name].append(episode[feature_name])
                            # put the number of participant to label
                            train_label_all_dic['Participant'].append(participant[0])
                elif participant[0] in test_P_list:
                    if ('bvp' in feature_name) or ('gsr' in feature_name) or ('skt' in feature_name):
                        if feature_name not in test_feature_all_dic.keys():
                            test_feature_all_dic[feature_name] = [episode[feature_name]]
                        else:
                            test_feature_all_dic[feature_name].append(episode[feature_name])
                    elif ('label' in feature_name):
                        if feature_name not in test_label_all_dic.keys():
                            test_label_all_dic[feature_name] = [episode[feature_name]]
                            # put the number of participant to label
                            test_label_all_dic['Participant'] = [participant[0]]
                        else:
                            test_label_all_dic[feature_name].append(episode[feature_name])
                            # put the number of participant to label
                            test_label_all_dic['Participant'].append(participant[0])
    # generate the ndarrar of features. For using feature-wise normalization
    train_feature_name_list = []
    train_feature_array = np.zeros(1)
    test_feature_name_list = []
    test_feature_array = np.zeros(1)
    for index, feature in enumerate(train_feature_all_dic.items()):
        if index == 0:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.array(feature[1])
        else:
            train_feature_name_list.append(feature[0])
            train_feature_array = np.vstack((train_feature_array, np.array(feature[1])))
    for index, feature in enumerate(test_feature_all_dic.items()):
        if index == 0:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.array(feature[1])
        else:
            test_feature_name_list.append(feature[0])
            test_feature_array = np.vstack((test_feature_array, np.array(feature[1])))
    # generate the label array.
    train_label_array = np.array(train_label_all_dic['label'])
    test_label_array = np.array(test_label_all_dic['label'])


    # get feature names
    train_P_label = train_label_all_dic['Participant']
    test_P_label = test_label_all_dic['Participant']
    # Transfer the P_labels to array
    train_P_label = np.array([int(i.replace('S', '')) for i in train_P_label])
    test_P_label = np.array([int(i.replace('S', '')) for i in test_P_label])
    # Remove the labels need to be ignored
    new_train_all_array = np.vstack((train_feature_array, train_label_array, train_P_label))
    new_test_all_array = np.vstack((test_feature_array, test_label_array, test_P_label))
    train_index = np.where((new_train_all_array[123, :] == 1) | (new_train_all_array[123, :] == 2)
                  | (new_train_all_array[123, :] == 3))
    test_index = np.where((new_test_all_array[123, :] == 1) | (new_test_all_array[123, :] == 2)
                  | (new_test_all_array[123, :] == 3))
    new_train_all_array = new_train_all_array[:, train_index[0]]
    new_test_all_array = new_test_all_array[:, test_index[0]]
    train_feature_array = new_train_all_array[:-2, :]
    train_label_array = new_train_all_array[-2, :]
    train_P_label = new_train_all_array[-1, :]
    test_feature_array = new_test_all_array[:-2, :]
    test_label_array = new_test_all_array[-2, :]
    test_P_label = new_test_all_array[-1, :]

    # To ensure the label is 0,1,2 and the training process will be correct, each label - 1.
    # The new label is: 0 - baseline, 1 - stress, 2 - amusement
    train_label_array = train_label_array - 1
    test_label_array = test_label_array - 1

    train_feature_array = train_feature_array.transpose()
    test_feature_array = test_feature_array.transpose()

    ## change nan to 0
    train_feature_array[np.isnan(train_feature_array)] = 0
    train_feature_array[np.isinf(train_feature_array)] = 0
    test_feature_array[np.isnan(test_feature_array)] = 0
    test_feature_array[np.isinf(test_feature_array)] = 0

    return train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label

def data_prepare_wesad(json_dic='json_files'):
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label = data_extraction_wesad(json_dic)

    ## Get the positions of Abnormal data. For analyzing the reasons
    # Transfer the P_labels to array
    # train_P_label = np.array([int(i.replace('S', '')) for i in train_P_label])
    # test_P_label = np.array([int(i.replace('S', '')) for i in test_P_label])
    # Stack all the train and test. Include labels and P numbers.
    train_feature_array = train_feature_array.transpose()
    test_feature_array = test_feature_array.transpose()
    train_array_all = np.vstack((train_feature_array, train_label_array, train_P_label))
    test_array_all = np.vstack((test_feature_array, test_label_array, test_P_label))
    # ignored in 5/6/7 this dataset
    train_array_all = train_array_all.transpose()
    test_array_all = test_array_all.transpose()
    train_del_list = []
    test_del_list = []
    for index, value in enumerate(train_array_all):
        if int(value[123]) >= 5:
            train_del_list.append(index)
    for index, value in enumerate(test_array_all):
        if int(value[123]) >= 5:
            test_del_list.append(index)
    train_array_all = np.delete(train_array_all, train_del_list, axis=0)
    test_array_all = np.delete(test_array_all, test_del_list, axis=0)

    train_array_all = train_array_all.transpose()
    test_array_all = test_array_all.transpose()
    return train_array_all, test_array_all, test_feature_name_list


def Abnormal_detection_getPosition(input_array):
    # This function for filter the abnormal data
    feature_array = input_array[:123, :]
    all_abnormal_data = []
    for index_f, feature in enumerate(feature_array):
        Percentile = np.percentile(feature, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        # If the data more than Uplimit or less than Downlimit, see it as abnormal data
        # Use the average between former one and later one to replace it
        for index in range(len(feature) - 1):
            if (feature[index] > UpLimit) or (feature[index] < DownLimit):
                try:
                    print("Up and Down:")
                    print(UpLimit)
                    print(DownLimit)
                    print("feature:")
                    print(feature[index - 3:index + 2])
                except:
                    continue

                abnormal_dic = {}
                abnormal_feature_num = index_f
                abnormal_feature_position = input_array[123:, index]
                abnormal_dic['Feature_num'] = abnormal_feature_num
                abnormal_dic['Feature_position'] = abnormal_feature_position
                all_abnormal_data.append(abnormal_dic)
    return all_abnormal_data


def generate_features_plot(input_array, feature_name, set_type):
    # This function for filter the abnormal data
    feature_array = input_array[:123, :]
    label_array = input_array[123:, :]
    x_label_all = []
    flag = 0
    now_P = 0
    for i in range(len(label_array[0])):
        if now_P != int(label_array[1][i]):
            flag = 1
            now_P = int(label_array[1][i])
        else:
            flag += 1
        x_label = str(int(label_array[1][i])) + '-' + str(flag) + '-' + str(
            int(label_array[0][i]))
        x_label_all.append(x_label)
    all_abnormal_data = []
    for index_f, feature in enumerate(feature_array):
        Percentile = np.percentile(feature, [0, 25, 50, 75, 100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3] + IQR * 1.5
        DownLimit = Percentile[1] - IQR * 1.5
        # If the data more than Uplimit or less than Downlimit, see it as abnormal data
        # Use the average between former one and later one to replace it
        for index in range(len(feature) - 1):
            if (feature[index] > UpLimit) or (feature[index] < DownLimit):
                abnormal_dic = {}
                abnormal_feature_num = index_f
                abnormal_feature_position = input_array[123:, index]
                abnormal_dic['Feature_num'] = abnormal_feature_num
                abnormal_dic['Feature_position'] = abnormal_feature_position
                all_abnormal_data.append(abnormal_dic)
        ## Draw plot of all date
        # plt.figure(dpi=50).set_figheight(15)
        # Distinguish the fear and no-fear
        colors = []
        plt.figure(figsize=(512, 10), dpi=60)
        for index, value in enumerate(x_label_all[::3]):
            value_list = value.split('-')
            if value_list[2] == '0':
                colors.append('blue')
                s0 = plt.scatter(x_label_all[::3][index], feature[::3][index], s=200, c='blue')
            elif value_list[2] == '1':
                colors.append('orange')
                s1 = plt.scatter(x_label_all[::3][index], feature[::3][index], s=200, c='orange')
            elif value_list[2] == '2':
                colors.append('green')
                s2 = plt.scatter(x_label_all[::3][index], feature[::3][index], s=200, c='green')
            elif value_list[2] == '3':
                colors.append('red')
                s3 = plt.scatter(x_label_all[::3][index], feature[::3][index], s=200, c='red')
            elif value_list[2] == '4':
                colors.append('gray')
                s4 = plt.scatter(x_label_all[::3][index], feature[::3][index], s=200, c='gray')
        # plt.figure(figsize=(512, 10), dpi=60)
        # plt.scatter(x_label_all[::3], feature[::3], s=200, c=colors)

        plt.legend((s0, s1, s2, s3, s4), ('not defined / transient', 'baseline', 'stress', 'amusement',
                                          'meditation'), loc="center left")
        plt.axhline(UpLimit, color='red')
        plt.axhline(DownLimit, color='red')
        plt.xticks(rotation=90)
        if set_type == 'test':
            plt.savefig('./test_features/' + feature_name[index_f] + '.jpg')
        if set_type == 'train':
            plt.savefig('./train_features/' + feature_name[index_f] + '.jpg')
        # plt.show()
        print(feature_name[index_f])
        print('Done')
        plt.close()


if __name__ == '__main__':
    train_array_all, test_array_all, test_feature_name_list = data_prepare_wesad(
        json_dic='json_files')
    # train_filter_array = Abnormal_detection_getPosition(train_array_all)
    # test_filter_array = Abnormal_detection_getPosition(test_array_all)
    generate_features_plot(test_array_all, test_feature_name_list, set_type='test')
    # generate_features_plot(train_array_all, test_feature_name_list, set_type='train')
