'''Train CIFAR10 with PyTorch.'''
import sys
import os
import time
sys.path.append("/home/junjiao/PycharmProjects/CNN_LSTM_emotion/")
from sklearn.metrics import f1_score

from Training.models.cnn_lstm import ParallelCNNLSTMModel, ResNetLSTM
from WESAD.Create_feature_maps_wesad_dimension import get_normalization_wesad_Dimension, \
    generate_feature_maps_wesad_Dimension


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import argparse
from models import *
from models import cnn_lstm
from Feature_wise_Normalization.Create_feature_maps_Dimension import generate_feature_maps_Dimension, \
    get_normalization_Dimension
from utils import progress_bar


def feature_map_trans(feature_maps):
    feature_maps_trans = np.zeros((feature_maps.shape[2], 1, feature_maps.shape[0], feature_maps.shape[1]))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, 0, :, :] = feature_map
    return feature_maps_trans


best_acc = 0  # best test accuracy
best_f1 = 0.0


def training_process(net_type, featureMap_stratgy, dimension_type):
    parser = argparse.ArgumentParser(description='PyTorch Empatia Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    best_f1 = 0.0

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    log_file = '../WESAD/log_data_normalization_WESAD_KNN.log'
    data_file = '../WESAD/json_files'

    train_array_all, test_array_all = get_normalization_wesad_Dimension(data_file, log_file)
    # Print train and test participants
    print('Train include: ' + str(set(train_array_all[124, :])))
    print('Test include: ' + str(set(test_array_all[124, :])))


    test_feature_map, test_label, test_arousal, test_valence = \
        generate_feature_maps_wesad_Dimension(test_array_all, strategy=featureMap_stratgy)
    #
    train_feature_map, train_label, train_arousal, train_valence = \
        generate_feature_maps_wesad_Dimension(train_array_all, strategy=featureMap_stratgy)

    if featureMap_stratgy == 'All_concat':
        test_feature_map1, test_label1, test_arousal1, test_valence1 = \
            generate_feature_maps_wesad_Dimension(test_array_all, strategy='AllFromOne')
        test_feature_map2, test_label2, test_arousal2, test_valence2 = \
            generate_feature_maps_wesad_Dimension(test_array_all, strategy='HalfAndHalf')
        test_feature_map3, test_label3, test_arousal3, test_valence3 = \
            generate_feature_maps_wesad_Dimension(test_array_all, strategy='HalfAndRandom')
        #
        train_feature_map1, train_label1, train_arousal1, train_valence1 = \
            generate_feature_maps_wesad_Dimension(train_array_all, strategy='AllFromOne')
        train_feature_map2, train_label2, train_arousal2, train_valence2 = \
            generate_feature_maps_wesad_Dimension(train_array_all, strategy='HalfAndHalf')
        train_feature_map3, train_label3, train_arousal3, train_valence3 = \
            generate_feature_maps_wesad_Dimension(train_array_all, strategy='HalfAndRandom')

        test_feature_map = np.dstack((test_feature_map1, test_feature_map2))
        test_feature_map = np.dstack((test_feature_map, test_feature_map3))
        test_label = np.hstack((test_label1, test_label2, test_label3))
        test_arousal = np.hstack((test_arousal1, test_arousal2, test_arousal3))
        test_valence = np.hstack((test_valence1, test_valence2, test_valence3))

        train_feature_map = np.dstack((train_feature_map1, train_feature_map2))
        train_feature_map = np.dstack((train_feature_map, train_feature_map3))
        train_label = np.hstack((train_label1, train_label2, train_label3))
        train_arousal = np.hstack((train_arousal1, train_arousal2, train_arousal3))
        train_valence = np.hstack((train_valence1, train_valence2, train_valence3))

    # decide the dimension type for training
    if dimension_type == 'label':
        nowTrain_label = train_label
        nowTest_label = test_label
    if dimension_type == 'arousal':
        nowTrain_label = train_arousal
        nowTest_label = test_arousal
    if dimension_type == 'valence':
        nowTrain_label = train_valence
        nowTest_label = test_valence

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map)
    x_train = feature_map_trans(train_feature_map)
    # Convert the ndarray to a PyTorch tensor
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(np.array(nowTrain_label).astype(int))

    # Create a TensorDataset object
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # Create a DataLoader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(np.array(nowTest_label).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Model CNN
    print('==> Building model..')

    net = net_type.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        print('\nTraining')
        # CNN definition
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            combine_input = inputs.squeeze(1)
            # combine_input = inputs.repeat(1, 3, 1, 1)

            outputs = net(combine_input)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        print('\nTesting')
        global best_acc
        global best_f1
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        prob_all = []
        label_all = []
        with torch.no_grad():
            time_list = []
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                combine_input = inputs.squeeze(1)
                # Test the time of processing a feature map
                start_time = time.time()
                outputs = net(combine_input)
                end_time = time.time()
                time_consumption = (end_time - start_time) * 1000
                time_list.append(time_consumption)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                prob = outputs.cpu().numpy()
                prob_all.extend(np.argmax(prob, axis=1))
                label = targets.cpu().numpy()
                label_all.extend(label)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print('mean time_consumption (ms): ' + str(np.mean(time_list)))
        # Calculate the F1 score
        f1 = f1_score(label_all, prob_all, average='weighted')
        # Save checkpoint.
        acc = 100. * correct / total
        F1 = 100. * f1
        print('F1:' + str(F1))
        if acc > best_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        if F1 > best_f1:
            best_f1 = F1
        print('Best ACC: ------- ' + str(best_acc))
        print('Best F1: ------- ' + str(best_f1))
        return acc, F1

    total_best = 0
    total_f1 = 0
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        now_acc, now_f1 = test(epoch)
        scheduler.step()
        if now_acc > total_best:
            total_best = now_acc
        if now_f1 > total_f1:
            total_f1 = now_f1
    return total_best, total_f1


if __name__ == '__main__':

    # Pick the model type
    # model_group = [VGG('VGG19'), ResNet18(), PreActResNet18(), GoogLeNet(), DenseNet121(), ResNeXt29_2x64d(),
    #                MobileNet(), MobileNetV2(), DPN92(), ShuffleNetG2(), SENet18(), ShuffleNetV2(1), EfficientNetB0(),
    #                RegNetX_200MF(), SimpleDLA()]
    # model_group = [VGG('VGG19'), ResNet18(), GoogLeNet(), DenseNet121(), EfficientNetB0(),
    #                MobileNet(), ResNeXt29_2x64d()]
    # model_group = [VGG('VGG19', num_classes=len(classes)), ResNet18(num_classes=len(classes)),
    #                GoogLeNet(num_classes=len(classes)), DenseNet121(num_classes=len(classes)),
    #                EfficientNetB0(num_classes=len(classes)),
    #                MobileNet(num_classes=len(classes)), ResNeXt29_2x64d(num_classes=len(classes))]
    input_size = 123
    hidden_size = 64
    num_layers = 2
    num_classes = 10
    Combine_model = ParallelCNNLSTMModel(input_size, hidden_size, num_layers, num_classes)
    # Combine_model = ResNetLSTM(input_size, hidden_size, num_layers, num_classes)

    dimension_types = ['label', 'arousal', 'valence']
    # dimension_types = ['arousal', 'valence']

    # Decide the feature maps generation strategy:
    # strategy_list = ['AllFromOne', 'HalfAndHalf', 'HalfAndRandom', 'All_concat']
    strategy_list = ['All_concat']
    for strategy in strategy_list:
        for dimension_type in dimension_types:
            # K-fold cross validation.
            K = 10
            bestAcc_list = []
            bestF1_list = []
            for cross_index in range(K):
                print('Start_dimension: ' + dimension_type)
                print('Start_strategy: ' + strategy)
                print('Start_Kfold: ' + str(cross_index + 1))
                best_acc = 0  # best test accuracy
                best_f1 = 0.0  # best test accuracy
                now_best, now_f1 = training_process(net_type=Combine_model, featureMap_stratgy=strategy,
                                                    dimension_type=dimension_type)
                bestAcc_list.append(now_best)
                bestF1_list.append(now_f1)
                print('ACC list:')
                print(bestAcc_list)
                print('F1 list:')
                print(bestF1_list)
            print('\n')
            print('dimension: ' + dimension_type)
            print('Strategy: ' + strategy)
            print('bestAcc_list: ')
            print(bestAcc_list)
            print('Acc std:' + str(np.std(bestAcc_list)))
            print('Mean best acc:' + str(np.mean(bestAcc_list)))
            print('bestF1_list: ')
            print(bestF1_list)
            print('F1 std:' + str(np.std(bestF1_list)))
            print('Mean best F1:' + str(np.mean(bestF1_list)))
