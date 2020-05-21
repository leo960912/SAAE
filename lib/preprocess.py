#how to load data?
#1. load the raw data, remove useless classes or select useful classes
#2. normalization the data(except label)
#3. add time window
#4.

import os
import numpy as np
from RootPATH import base_dir

def remove_specific_class(input, useless_type):
    '''
    remove some data with label useless_type
    :param input: an array shaped (N, D)
    :param useless_type: an integer specify the useless label
    :return:
    '''
    return input[input[:, -1]!=useless_type, :]

def choose_specific_classes(input, class_types, class_num=13):
    '''
    choose the data belong to only specified classes
    :param input: an array shaped (N, D)
    :param class_types: a list contained class labels
    :param max_class: number of all classes
    :return: an array
    '''
    class_types = int(np.array(class_types))
    mask = np.zeros((class_num,), dtype=bool)
    mask[class_types] = True
    return input[mask[input[:, -1].astype(int)], :]


#50HZ, 1s is 20 samples, overlap 50%
def add_window(input, window_size=20, overlap=0.5):
    '''
    add sliding window to the input data
    :param input: an array shaped (N, D)
    :param window_size: an integer
    :param overlap: a number range from 0 to 1
    :return:
    '''
    length = input.shape[0]
    step = int(window_size*(1-overlap))
    output = []
    for i in range(0,length-window_size+1,step):
        output.append(input[i:i+window_size, :])
    return np.stack(output, axis=0)

#upsampling or downsampling

#minmax or normlization


def one_hot(labels):
    '''
    one hot encoding the labels
    :param labels: a 1D array containing labels
    :return: a 2D array
    '''
    labels = np.squeeze(labels)
    labels = labels.astype('int')
    max = labels.max()
    min = labels.min()
    length = len(labels)

    if max-min == max:
        onehot_label = np.zeros((length, max+1))
        onehot_label[np.arange(length), labels] = 1
        return onehot_label
    elif min == 1:
        onehot_label = np.zeros((length, max))
        onehot_label[np.arange(length), labels-1] = 1
        return onehot_label

def map_label(act_id, dataset):
    label_list = []
    if dataset == 'MHEALTH':
        label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif dataset == 'PAMAP2':
        label_list = [1,  2,  3,  4,  5,  6,  7, 12, 13, 16, 17, 24]
    elif dataset == 'USCHAD':
        label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif dataset == 'UCIDSADS':
        label_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    else:
        return act_id
    label = label_list.index(int(act_id))
    return label

#label
def extract_label(input, dataset='MHEALTH'):
    '''
    process the label in each window
    :param input: 3-D array [N, window_size, D]
    :return:
    '''
    length = input.shape[0]
    masks = []
    labels = []
    for i in range(0, length):
        label = input[i, :, -1]
        identical = all(x==label[0] for x in label)
        masks.append(identical)
        if identical == True:
            labels.append(map_label(label[0], dataset))
    features = input[masks, :, 0:-1]
    #if onehot:
        #labels = one_hot(np.array(labels))
    #else:
        # minors 1 because labels start from 1
    labels = np.array(labels).reshape(-1, 1)
    return features, labels

if __name__ == '__main__':

    MHEALTH_DATA_FILES = ['mHealth_subject' + str(i) + '.log' for i in range(1, 11)]
    DATA_PATH = [os.path.join(base_dir, 'data','MHEALTHDATASET', i) for i in MHEALTH_DATA_FILES]

    subject = np.loadtxt('../data/MHEALTHDATASET/mHealth_subject9.log', dtype=float, delimiter='\t')
    print(subject.shape, subject[0, 1], subject.max(), subject.min())
    label = subject[:, -1]
    print(label.shape, label.dtype, label.max(), label.min())
    subject = remove_specific_class(subject, 0)
    #subject1 = choose_specific_classes(subject1, [0], class_num=13)
    print(subject.shape)
    feature = subject[:, 0:-1]
    print(feature.dtype)
    feature = feature.astype(float)
    zeros = np.zeros_like(feature[0, :])
    count = 0
    print(zeros.shape, feature[0, :].shape, zeros.dtype, feature.dtype)
    for i in range(subject.shape[0]):
        if np.array_equal(feature[i, :], zeros):
            count = count + 1
    print(count)
    #mask = np.zeros((12,), dtype=bool)
    #mask[[0]] = True
    #out = mask[label.astype(int)]

    '''
    print('original: ', subject1.shape)
    #original labels for 0 to 12, 0 means nothing
    #print(subject1[:,-1].max(), subject1[:,-1].min())
    subject1 = remove_specific_class(subject1, 0)
    #78.67% of the data belongs to class 0
    print(subject1.shape)
    subject1 = add_window(subject1)
    print(subject1.shape)
    features, labels = add_label(subject1)
    '''
