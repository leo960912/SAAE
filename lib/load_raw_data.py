import os
import numpy as np
import time
from scipy.io import arff
from io import StringIO
import pandas as pd
import scipy.io as scio
def load_one_USCHAD(id, dir):
    sub_dir = 'Subject' + str(id)
    path = os.path.join(dir, 'data', 'USCHAD', sub_dir)
    print(path)
    data = np.array([]).reshape(0, 7)
    for trial in range(1, 6):
        for act in range(1, 13):
            print(trial,act)
            file = 'a{}t{}.mat'.format(act, trial)
            tri_data = scipy.io.loadmat(os.path.join(path, file))
            """
            if tri_data['subject'] != [str(id)]:
                print('The subject id do not match')
                continue
            if tri_data['activity_number'] != [str(act)]:
                print('The activity number do not match', act, trial, tri_data['activity_number'])
                continue
            if tri_data['trial'] != [str(trial)]:
                print('The trail number do not match', act, trial, tri_data['trial'])
                continue
            """
            tri_data = tri_data['sensor_readings']
            #print(tri_data['sensor_readings'].shape)
            tri_label = np.array([act] * len(tri_data)).reshape(-1, 1)
            #print(tri_data.shape, tri_label.shape)
            current_data = np.concatenate((tri_data, tri_label), axis=1)
            data = np.concatenate((data, current_data), axis=0)
    return data

def load_one_UCIDSADS(dir, id):
    all_data = []
    path = os.path.join(dir, 'data', 'UCIDSADS')
    for act in range(1, 20):
        for seg in range(1, 61):
            if act < 10:
                act_id = '0' + str(act)
            else:
                act_id = str(act)
            if seg < 10:
                seg_id = '0' + str(seg)
            else:
                seg_id = str(seg)
            sub_dir = os.path.join('a' + act_id, 'p' + str(id), 's' + seg_id + '.txt')
            data = np.loadtxt(os.path.join(path, sub_dir), delimiter=',')
            label = np.array([act] * len(data)).reshape(-1, 1)
            data = np.concatenate((data, label), axis=1)
            all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    return all_data

def load_one_WISDM(dir, id):
    file_name = os.path.join(dir, 'data', 'WISDM', 'WISDM_ar_v1.1_transformed.arff')
    f = StringIO(open(file_name,'r').read())
    data,meta=arff.loadarff(f)
    d = pd.DataFrame(data)
    d['user'] = d['user'].apply(lambda x: int(x))
    cls_dict = {}
    #{b'Jogging': 0, b'Walking': 1, b'Upstairs': 2, b'Downstairs': 3, b'Sitting': 4, b'Standing': 5}

    for i in range(len(d['class'].unique())):
        cls_dict[d['class'].unique()[i]] = i
    d['class'] = d['class'].apply(lambda x : cls_dict[x])
    feature = d.columns.drop(['UNIQUE_ID','user','class'])
    feature_data = d[feature][d['user']==id].values
    label_data = d['class'][d['user']==id].values[:,np.newaxis]
    all_data = np.concatenate((feature_data,label_data),axis=1)
    return all_data

def load_one_OPPO(dir, id):
    data = []
    #0 drill 1-5 normal
    for _ in range(6):
        data.append(np.load(os.path.join(dir, 'data', 'Opportunity', 'subject{}_{}.npy'.format(id,_))))
    all_data = np.concatenate(data,0)
    return all_data


def load_one_ActRec(dir, id):
    data_file_name = os.path.join(dir, 'data', 'ActRecTut-master', 'Data', 'subject{}_gesture'.format(id), 'data.mat')
    #label_file_name = os.path.join(dir, 'data', 'ActRecTut-master', 'data', 'subject{}_gesture'.format(id,_), 'labels.dat')
    data = scio.loadmat(data_file_name)
    feature = data['data']
    labels = data['labels']
    all_data = np.concatenate([feature,labels],1)
    return all_data

def load_one_raw_data(name, dir, id):
    '''
    load raw data from the specified dictionary
    :param name: name of the dataset
    :param dir: location
    :param ids:  a list contains users id to be loaded
    :return: a numpy array
    '''
    raw_data = None
    #for user in ids:
    if name == 'MHEALTH':
        if id not in range(1, 11):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'MHEALTHDATASET', 'mHealth_subject{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_file = os.path.join(dir, 'data', 'MHEALTHDATASET', 'mHealth_subject{}.log'.format(id))
            print('Load raw data from:', raw_file)
            raw_data = np.loadtxt(raw_file, dtype=float, delimiter='\t')
            raw_data = raw_data[raw_data[:, -1] != 0, :]
            np.save(file_name, raw_data)
    elif name == 'PAMAP2':
        if id not in range(1, 9):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'PAMAP2', 'subject10{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_file = os.path.join(dir, 'data', 'PAMAP2', 'subject10{}.dat'.format(id))
            print('Load raw data from:', raw_file)
            raw_data = np.loadtxt(raw_file)
            mask = [False, True, False] + ([False] + [True] * 4 * 3 + [False] * 4) * 3
            raw_data = raw_data[:, mask]
            raw_data = np.hstack((raw_data[:, 1:], raw_data[:, 0:1]))
            raw_data = raw_data[raw_data[:, -1] != 0, :]
            np.save(file_name, raw_data)
    elif name == 'USCHAD':
        if id not in range(1, 15):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'USCHAD', 'subject{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_data = load_one_USCHAD(id, dir)
            np.save(file_name, raw_data)
    elif name == 'UCIDSADS':
        if id not in range(1, 9):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'UCIDSADS', 'subject{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_data = load_one_UCIDSADS(dir, id)
            np.save(file_name, raw_data)
    elif name == 'WISDM':
        if id not in range(1, 37):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'WISDM', 'subject{}.npy'.format(id))
        
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_data = load_one_WISDM(dir, id)
            np.save(file_name, raw_data)
    elif name == 'OPPO':
        if id not in range(1, 4):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'Opportunity', 'subject{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_data = load_one_OPPO(dir, id)
            np.save(file_name, raw_data)
    elif name == 'ActRec':
        if id not in range(1, 3):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'ActRecTut-master', 'subject{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raw_data = load_one_ActRec(dir, id)
            np.save(file_name, raw_data)
    elif name == 'EEG':
        if id not in range(1, 51):
            raise Exception('User is not included!')
        file_name = os.path.join(dir, 'data', 'EEG', '{}.npy'.format(id))
        if os.path.exists(file_name):
            print('Load raw data from:', file_name)
            raw_data = np.load(file_name)
        else:
            raise('file not exits')
    elif name == 'HAR':
        """here"""
    else:
        raise ValueError
    return raw_data

if __name__ == '__main__':
    #from RootPATH import base_dir
    #data = load_one_USCHAD(id=1, dir=base_dir)
    #data = load_one_UCIDSADS(dir=base_dir, id=1)
    #print(data.shape)

    start_time = time.time()
    for i in range(1, 4):
        data = load_one_raw_data('OPPO', '..\\', i)
        print(data.shape)
        labels, count = np.unique(data[:, -1], return_counts=True)
        print(labels, count)
    print("Total time cost: %s seconds ---" % (time.time() - start_time))



