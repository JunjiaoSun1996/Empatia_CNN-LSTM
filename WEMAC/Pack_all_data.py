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
@Time : Created in 3:55 PM 2024/02/19   
@FileName: Pack_all_data.py                           
@Software: PyCharm


Introduction of this File:
When we have designed the 123 feature extraction functions already, we need to generate all of the features into a
json file to save time. This file is created for generating those features into a stable file.
"""

import json
import os

import numpy as np
import scipy.io
import mat73
import traceback
from SOP_del import SOP_del
from BVP_Signal_Features import BVP_features
from GSR_Signal_Features import GSR_features
from SKT_Signal_Features import SKT_features
import pandas as pd

def get_label(participant_num, video_num, feature_dic):
    ## for put all the label in Labels_TabLab_IT07 to dictionary. Every line of data has its label.
    participant_num = participant_num.replace('P', '')
    # import the data of label
    file_path = r'./Labels_TabLab_IT07.xls'
    labal_data = pd.read_excel(file_path, header=0)
    labal_data = labal_data.to_numpy()
    feature_dict_withLabel = feature_dic
    for i in range(len(labal_data)):
        if labal_data[i][0] == int(participant_num) and labal_data[i][1] == int(float(video_num)):
            feature_dict_withLabel['Tanda'] = labal_data[i][2]
            feature_dict_withLabel['ReportadaBinarizado'] = labal_data[i][3]
            feature_dict_withLabel['TargetBinarizado'] = labal_data[i][4]
            feature_dict_withLabel['PADBinarizado'] = labal_data[i][5]
            feature_dict_withLabel['Arousal'] = labal_data[i][6]
            feature_dict_withLabel['Valencia'] = labal_data[i][7]
            feature_dict_withLabel['Dominancia'] = labal_data[i][8]
    return feature_dict_withLabel

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

if __name__ == '__main__':
    data_dict = mat73.loadmat('./BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    feature_dict_all = {}
    # Parameters for features
    len_frames = 3000
    overlap_frames = 2500
    bvp_sample_rate = 200
    gsr_sample_rate = 200
    skt_sample_rate = 10
    start_index = 0
    ## Loop for getting data
    for participant in data_list:
        feature_dict_p = {}
        for video in participant:
            feature_dict_v = {}
            try:
                ## BVP
                bvp_data = video['EH']['Video']['raw']['bvp_filt']
                ## GSR
                gsr_data = video['EH']['Video']['raw']['gsr_uS_filtered_sm']
                ## SKT
                skt_data = video['EH']['Video']['raw']['skt_filt_dn_sm']
                while (start_index + len_frames) < len(bvp_data):
                    # For getting the data, define some parameters for SOP
                    bvp_data_sample = bvp_data[start_index:start_index + len_frames]
                    # Calculate SOP for BVP features
                    Slope_list, Onset_list, Peak_list = SOP_del(bvp_data_sample, bvp_sample_rate)
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
                    skt_data_sample = skt_data[int(start_index / 20):int((start_index + len_frames) / 20)]
                    # define signal is raw_data_sample, frequency is 20
                    skt_features = SKT_features(signal=skt_data_sample, sampling_rate=skt_sample_rate)
                    # get skt features_dict
                    skt_features_dict = skt_features.SKT_get_features_dic()

                    feature_all_data_sample = dict(bvp_features_dict, **gsr_features_dict)
                    feature_all_data_sample = dict(feature_all_data_sample, **skt_features_dict)
                    ## Put the label in the dic
                    feature_dict_withLabel = get_label(participant_num=participant[0]['ParticipantNum'],
                                                       video_num=str(video['Trial']), feature_dic=feature_all_data_sample)
                    feature_dict_v[str(video['Trial']) + '[' + str(start_index) + ':' +
                                   str(start_index + len_frames) + ']'] = feature_dict_withLabel
                    print("Participant number:" + str(participant[0]['ParticipantNum']))
                    print("Video episode done:" + str(video['Trial']) + '[' + str(start_index) + ':' +
                                   str(start_index + len_frames) + ']')
                    start_index += (len_frames - overlap_frames)
                feature_dict_p[str(video['Trial'])] = feature_dict_v
                start_index = 0
                print("Participant number:" + str(participant[0]['ParticipantNum']))
                print("Video done:" + str(video['Trial']))
            except Exception as e:
                print(e.args)
                print('========================')
                print(traceback.format_exc())
                print(participant[0]['ParticipantNum'] + ' Done')
                continue
        feature_dict_all[participant[0]['ParticipantNum']] = feature_dict_p
        print("Participant done:" + str(participant[0]['ParticipantNum']))
    for key, value in feature_dict_all.items():
        print('{key}:{value}'.format(key=key, value=value))

    path = "./Pack_all_data.json"
    s = SaveJson()
    s.save_file(path, feature_dict_all)
