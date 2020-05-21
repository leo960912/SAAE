import os
import torch
import torch.utils.data
import numpy as np
import lib.preprocess as preprocess
from lib.load_raw_data import load_one_raw_data
from lib.normalizaiton import minmax_by_column, standard_by_column
from RootPATH import base_dir

'''
#previous version
def load_one_subject(user, path, window=20, overlap=0.5, encoding=False, normalize=False):
    #load one subject's data
    #:param user: subject ID, an int
    #:param path: data path, a list
    #:return: two 2D numpy array
    if user not in range(1, 11):
        raise Exception('User is not included!')
    data = np.loadtxt(path[user-1], dtype=float, delimiter='\t')
    data = preprocess.remove_specific_class(data, 0)
    #data = load_one_raw_data(dataset, path, user)
    data = preprocess.add_window(data, window_size=window, overlap=overlap)
    features, labels = preprocess.extract_label(data, onehot=encoding)
    if normalize == True:
        #features = minmax_by_column(features)
        features = standard_by_column(features)
    return features, labels
'''
def get_minimum_len(dataset):
    if dataset == 'MHEALTH':
        return 300
    elif dataset == 'PAMAP2':
        #or maybe 1500
        return 2000
    elif dataset == 'USCHAD':
        return 1000
    else:
        raise ValueError
#new version
def load_one_subject(user, path, dataset='MHEALTH', window=20, overlap=0.5,
                     encoding=False, normalize=False, drop_redundant=False):
    #load one subject's data
    #:param user: subject ID, an int
    #:param path: data path, a list
    #:return: two 2D numpy array
    data = load_one_raw_data(dataset, path, user)
    #remove rows that contain non
    data = data[~np.isnan(data).any(axis=1)]
    data = preprocess.add_window(data, window_size=window, overlap=overlap)
    features, labels = preprocess.extract_label(data, dataset=dataset)
    if drop_redundant == True:
        minimum_len = get_minimum_len(dataset)
        features, labels = drop_redundant_data(features, labels, minimum_len)
    if encoding == True:
        labels = preprocess.one_hot(labels)
    if normalize == True:
        #features = minmax_by_column(features)
        features = standard_by_column(features)
    return features, labels

def load_multiple_subjects(users, path, dataset='MHEALTH', window=20, overlap=0.5,
                           encoding=False, normalize=False, drop_redundant=False):
    '''
    load multiple user's data
    :param users: a list contains user ids
    :param path: data path, a list
    :return: two 2D numpy array
    '''
    features, labels = [], []
    for i in users:
        feature, label = load_one_subject(i, path=path, dataset=dataset, window=window,
                                          overlap=overlap, encoding=encoding, normalize=normalize,
                                          drop_redundant=drop_redundant)
        features.append(feature)
        labels.append(label)
    return np.vstack(features), np.vstack(labels)

def drop_redundant_data(x, y, minimum_len=1000):
    #x and y are numpy arrays
    assert len(x) == len(y)
    x_ = np.array([]).reshape((0, x.shape[1], x.shape[2]))
    y_ = np.array([]).reshape((0, 1))
    labels, count = np.unique(y, return_counts=True)
    #minimum_len = count.min()
    for label in labels:
        label_idx = np.argwhere(np.squeeze(y) == label).reshape(-1)
        if len(label_idx) > minimum_len:
            index = np.arange(len(label_idx))
            np.random.shuffle(index)
            label_idx = label_idx[index[0:minimum_len]]
        x_ = np.concatenate((x_, x[label_idx]), axis=0)
        y_ = np.concatenate((y_, y[label_idx]), axis=0)
    return x_, y_

def split_train_validation(features, labels, ratio=0.8):
    assert len(features) == len(labels), 'inputs do not match!'
    indics = np.random.permutation(len(features))
    split = int(len(features) * ratio)
    train_idx, val_idx = indics[:split], indics[split:]
    train_fea, val_fea = features[train_idx], features[val_idx]
    train_label, val_label = labels[train_idx], labels[val_idx]
    return train_fea, train_label, val_fea, val_label

def get_dataloader(features, labels, batch_size, mode='train', flag = True):
    cuda = True if torch.cuda.is_available() else False
    cuda = True if flag else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorInt = torch.cuda.LongTensor if cuda else torch.LongTensor
    X = TensorFloat(features)
    Y = TensorInt(labels)
    data = torch.utils.data.TensorDataset(X, Y)
    #train_length = int(len(data) * 0.8)
    #test_length = int(len(data)) - train_length
    #train_data, test_data = torch.utils.data.random_split(data, [train_length, test_length])
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader
"""
if __name__ == '__main__':
    # MHEALTH subject 10, activity 12
    from RootPATH import base_dir
    #test load_one_subject
    for i in range(1, 11):
        subject1_x, subject1_y = load_one_subject(i, base_dir, dataset='MHEALTH',
                                                  encoding=False, drop_redundant=False)
        print(subject1_x.shape, subject1_y.shape)
        #print(subject1_x.max(), subject1_x.min())
        #print(subject1_x.mean(), subject1_x.std())
        #print(subject1_y.max(), subject1_y.min())
        labels, count = np.unique(subject1_y, return_counts=True)
        print(labels, count, count.min())
"""
'''
#test new load 
x, y = load_one_subject(1, base_dir, dataset='USCHAD',
                        encoding=False, normalize=False, drop_redundant=True)
print(x.shape, y.shape)
x, y = load_one_subject(1, base_dir, dataset='USCHAD',
                        encoding=True, normalize=False, drop_redundant=False)
print(x.shape, y.shape)
x, y = load_one_subject(1, base_dir, dataset='USCHAD',
                        encoding=True, normalize=False, drop_redundant=True)
print(x.shape, y.shape)
'''

'''
#test drop_redundant_data
print(x.shape, y.shape)
labels, count = np.unique(y, return_counts=True)
print(labels, count)
x, y = drop_redundant_data(x, y)
print(x.shape, y.shape)
'''

'''
MHEALTH_DATA_FILES = ['mHealth_subject' + str(i) + '.log' for i in range(1, 11)]
DATA_PATH = [os.path.join(base_dir, 'data', 'MHEALTHDATASET', i) for i in MHEALTH_DATA_FILES]
for i in range(1, 11):
    print(i)
    if int(i) not in range(1, 11):
        raise Exception('User is not included!')
    subject1_x, subject1_y = load_one_subject(i, DATA_PATH, normalize=True)
    print(subject1_x.shape, subject1_y.shape)
'''

'''
print(subject1_x.shape, subject1_y.shape, subject1_x.max(), subject1_x.min())
zeros = np.zeros_like(subject1_x[0, :])
count = 0
for i in range(subject1_x.shape[0]):
    if np.array_equal(subject1_x[i, :], zeros):
        count = count +1
print(count)
'''

'''
#load_one_subject API
subject1_x, subject1_y = load_one_subject(1, DATA_PATH)
#data shape [Number, Window, Feature_nums]
train_fea, train_label, val_fea, val_label = split_train_validation(subject1_x, subject1_y)
print(train_fea.shape, train_label.shape, val_fea.shape, val_label.shape)
train_loader = get_dataloader(subject1_x, subject1_y, 32)
for i, j in train_loader:
    print(i.shape, j.shape)
    exit()
'''

#load_multiple_subject API
'''
subject1_x, subject1_y = load_one_subject(1, DATA_PATH)
print(subject1_x.shape, subject1_y.shape)
subject2_x, subject2_y = load_one_subject(2, DATA_PATH)
print(subject2_x.shape, subject2_y.shape)
subject3_x, subject3_y = load_one_subject(3, DATA_PATH)
print(subject3_x.shape, subject3_y.shape)
subjects_x, subjects_y = load_multiple_subjects([1, 2, 3], DATA_PATH)
print(subjects_x.shape, subjects_y.shape)
'''



