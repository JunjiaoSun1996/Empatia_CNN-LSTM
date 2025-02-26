# Empatia_CNN-LSTM (Project in CEI-UPM)

This project is for the Python code for Empatia based on deep learning. The DL model for negative recognition is a conbination of CNN and LSTM. Using WEMAC and WESAD data base.

The paper related to this project has been published in **DOI**: 10.1109/BIBM62325.2024.10822762

**Please, pay attention to all of the file locations in the project. Change them to yours. That is the most impotant thing.**

Environment GPU: Nvidia A30 Cuda: 12.2 Driver: 535.154.05 Torch: 2.0.0+cu118

**WEMAC:**

1.BVP_Signal_Features, GSR_Signal_Features and SKT_Signal_Features.py are the python files for calculating all of the 123 physiological features. 

2.Pack_all_data.py used for generating a json file to save those 123 features because it would be a huge time consumer if we call the WEMAC matlab data set every time. Remember change the location of .mat file and label file. If you want to use IT06.mat, please use Transfer_label.py firstly. 

3.Data_normalization.py created for feature normalization because the calculation functions have different data level. We use FWN to normalize them and generate a .log for recording which normalization function should be used for a specific feature. Please change the name of .log. 

4.Create_feature_maps.py includes the method how to generate 2D feature maps based on the json file and FWN log file.

5.Then we could start to train those feature maps using CNN-LSTM. In Training/WEMAC_cnn_lstm_it06.py, we tested the model performance without fueature fusion. In Training/WEMAC_cnn_lstm_feature_fusion_it06.py, we tested the performance with feature fusion.

**WESAD** 

1.The feature calculation methods are just same as WEMAC. 

2.Feature_extraction_wesad.py record the methods for writing all of the features into several json files. Please check the root. You can download WESAD here https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection 

3.Data_normalization.py is just similar to WEMAC. The output is a .log file. 

4.Creat_feature_maps_wesad.py could generate 2D feature maps using in training process. 

5.Then we can start the series of training process. We have the same training files with WEMAC.

**PLEASE CITE:**

@inproceedings{sun2024negative,
  title={Negative emotion recognition based on physiological signals using a CNN-LSTM model},
  author={Sun, Junjiao and Portilla, Jorge and Otero, Andres},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3736--3741},
  year={2024},
  organization={IEEE Computer Society}
}
