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
@Time : Created in 5:41 PM 2024/02/19   
@FileName: Feature_extraction_wesad.py                           
@Software: PyCharm


Introduction of this File:
This file is created for extracting the 123 features from WESAD. And pack them into several json files.
"""

import json
import os

import numpy as np
from collections import Counter
from read_data import chest_data_label
from WESAD.BVP_Signal_Features import BVP_features
from WESAD.GSR_Signal_Features import GSR_features
from WESAD.SKT_Signal_Features import SKT_features
from SOP_del_wesad import SOP_del_wesad


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class SaveJson(object):

    def save_file(self, path, item):
        item = json.dumps(item, cls=MyEncoder)
        try:
            if not os.path.exists(path):
                with open(path, "w", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
            else:
                with open(path, "a", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)


def wesad_features(path_cwd):
    subs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # subs = [2, 3, 4]
    # check the data has been generated
    files = os.listdir(path_cwd)
    # for file in files:
    #     file_list = file.split('_')
    #     file_num = int(file_list[-1].replace('.json', '').replace('S', ''))
    #     if file_num in subs:
    #         subs.remove(file_num)

    # chest_data_all, labels = chest_data_label('C:\\Users\\JUNJIAO SUN\\PycharmProjects\\Empatia\\WESAD\\WESAD', subs=subs)
    chest_data_all, labels = chest_data_label('/home/junjiao/PycharmProjects/Empatia/WESAD/WESAD', subs=subs)

    all_subject_dic = {}
    # Parameters for features
    len_frames = 6000
    overlap_frames = 4000
    bvp_sample_rate = 700
    gsr_sample_rate = 700
    skt_sample_rate = 700
    start_index = 0
    for subject in chest_data_all.keys():
        feature_dict_subject = {}
        # In wesad, it does not have BVP. So we use ecg.
        bvp_data = np.ravel(chest_data_all[subject]['ECG'])
        gsr_data = np.ravel(chest_data_all[subject]['EDA'])
        skt_data = np.ravel(chest_data_all[subject]['Temp'])
        # get the label of subject data
        label = labels[subject]
        while (start_index + len_frames) < len(bvp_data):
            # For getting the data, define some parameters for SOP
            bvp_data_sample = bvp_data[start_index:start_index + len_frames]
            # Calculate SOP for BVP features
            Slope_list, Onset_list, Peak_list = SOP_del_wesad(bvp_data_sample, bvp_sample_rate)
            # define signal is raw_data_sample, frequency is 200
            bvp_features = BVP_features(signal=bvp_data_sample, slope_list=Slope_list, onset_list=Onset_list,
                                        peak_list=Peak_list, sampling_rate=bvp_sample_rate)
            # get bvp features_dict
            bvp_features_dict = bvp_features.BVP_get_features_dic()

            ## For gsr data
            gsr_data_sample = gsr_data[start_index:start_index + len_frames]
            # define signal is raw_data_sample, frequency is 200
            gsr_features = GSR_features(signal=gsr_data_sample, sampling_rate=gsr_sample_rate)
            # get gsr features_dict
            gsr_features_dict = gsr_features.GSR_get_features_dic()

            ## For skt data
            # Because frequency in skt is 10hz. So the length of the data should be
            # divided by bvp_rate / skt_rate
            skt_data_sample = skt_data[start_index:start_index + len_frames]
            # define signal is raw_data_sample, frequency is 20
            skt_features = SKT_features(signal=skt_data_sample, sampling_rate=skt_sample_rate)
            # get skt features_dict
            skt_features_dict = skt_features.SKT_get_features_dic()

            feature_all_data_sample = dict(bvp_features_dict, **gsr_features_dict)
            feature_all_data_sample = dict(feature_all_data_sample, **skt_features_dict)
            # calculate the label of this episode
            label_sample = label[start_index:start_index + len_frames]
            label_episode = Counter(label_sample).most_common(1)[0][0]
            print('Label episode: ' + str(label_episode))

            feature_all_data_sample['label'] = label_episode
            feature_dict_subject['Episode' + '[' + str(start_index) + ':' +
                                 str(start_index + len_frames) + ']'] = feature_all_data_sample
            print("Subject number:" + subject)
            print("Subject episode done:" + '[' + str(start_index) + ':' +
                  str(start_index + len_frames) + ']')
            start_index += (len_frames - overlap_frames)
        all_subject_dic[subject] = feature_dict_subject
        start_index = 0
        print("Subject done:" + subject)
        # write the subject data into json
        path = path_cwd + '\\Pack_all_data_WESAD_' + str(subject) + '.json'
        s = SaveJson()
        s.save_file(path, feature_dict_subject)

    return all_subject_dic


if __name__ == '__main__':
    root_path = os.path.join(os.getcwd(), 'json_files')
    path = root_path + '\\Pack_all_data_WESAD.json'
    all_subject_dic = wesad_features(path_cwd=root_path)

    # Integrate all json
    feature_dic_all = {}
    for (root, dirs, files) in os.walk(root_path):
        for file in files:
            file_list = file.split('_')
            file_num = file_list[-1].replace('.json', '')
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                content = json.load(f)
            feature_dic_all[file_num] = content
    s = SaveJson()
    s.save_file(path, feature_dic_all)
    print(1)
