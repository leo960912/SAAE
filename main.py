# -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack,signal
import sklearn.preprocessing as pre
import pandas as pd
from DAE import weights_init_normal, Score, Encoder, Decoder, win_fft, Discriminator, MCCNN, pMCCNN, deMCCNN, DD
import torch
from scipy.signal import butter, lfilter
import os
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.nn.functional as F
from lib.init_weights import init_weights
from lib.data_loader import load_one_subject, load_multiple_subjects, split_train_validation, get_dataloader
import random
from multiprocessing import Pool

base_dir = ''

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def save_set(it,res,res_set,num):
    cat = ['precision', 'recall', 'f1-score']
    temp = []
    #for i in range(num):
    #    for j in cat:
    #        temp.append(res[str(i)][j])
    for i in ['micro avg','macro avg', 'weighted avg']:
        for j in cat:
            temp.append(res[i][j])
    res_set.append([it,temp])
    return res_set
def build(columns,res_set):
    result = np.array([i[1] for i in res_set])
    index = np.array([i[0] for i in res_set])
    return pd.DataFrame(result,index=index, columns=columns)
def extract(input, n_fea, time_window, moving):
    global n_classes 
    n_classes = len(set(input[:,-1]))
    xx = input[:, :n_fea]
    """Filtering"""
    # data_f1 = []
    # #  EEG Delta pattern decomposition
    # for i in range(xx.shape[1]):
    #     x = xx[:, i]
    #     fs = 250.0
    #     lowcut = 0.5
    #     highcut = 100
    #
    #     y = butter_bandpass_filter(x, lowcut, highcut, fs, order=4)
    #     data_f1.append(y)
    # xx = np.transpose(np.array(data_f1))

    # xx = preprocessing.scale(xx)  # z-score normalization
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = (xx.shape[0] - time_window) // moving + 1
    for i in range(number):
        ave_y = list(set(list(yy[(i * moving):(i * moving + time_window)].squeeze())))
        if len(ave_y) == 1:
            new_x.append(xx[(i * moving):(i * moving + time_window), :])
            new_y.append(ave_y[0])

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea, time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    return new_x, new_y

        
class experiment(object):
    def __init__(self, data, small_batch, multi_channel, ratio, name, id___):
        self.data = data
        self.name = name
        self.batch = small_batch
        self.multi_channel = multi_channel
        self.ratio = ratio
        self.id___ = id___
        if self.multi_channel:
            self.resume(ratio = self.ratio)
        
        """to here"""
        self.setTrain()
    
    def resume(self, ratio = 0.2):
        self.num = int(self.data.inner_identity.shape[2] * ratio)
        self.in_sample = np.arange(self.data.inner_identity.shape[2])
        temp_ = pd.DataFrame(self.data.person,columns=['person'])
        temp_['channel'] = np.tile(0, self.data.data.shape[0])
        temp_['data'] = [self.data.data[j,:,:] for j in range(self.data.data.shape[0])]
        temp_['inner'] = [np.mean(self.data.inner_identity[j,:,:].squeeze(),0) for j in range(self.data.data.shape[0])]
        temp_['outter'] = [np.mean(self.data.outter_identity[j,:,:].squeeze(),0) for j in range(self.data.data.shape[0])]
        temp_['frequency domain'] = [np.mean(self.data.raw[j,:,:].squeeze(),0) for j in range(self.data.data.shape[0])]
        temp_['inner_up'] = temp_['inner'].apply(lambda x: np.mean(x[x.argsort()][-self.num:]),1)
        temp_['inner_down'] = temp_['inner'].apply(lambda x: np.mean(x[x.argsort()][:self.data.inner_identity.shape[2]-self.num]),1)
        temp_['inner_up_loc'] = temp_['inner'].apply(lambda x: list(self.in_sample[x.argsort()][-self.num:]),1)
        temp_['label'] = self.data.label
        temp_.index = range(len(temp_))
        
        temp_['inner_down_loc'] = temp_['inner_up_loc'].apply(lambda x: self.down_loc([x],self.data.inner_identity.shape[2])[0],1)
        temp_['inner_up_loc'] = temp_['inner_up_loc'].apply(lambda x : self.onehot(x,self.data.inner_identity.shape[2]),1)
        temp_['inner_down_loc'] = temp_['inner_down_loc'].apply(lambda x: self.onehot(x,self.data.inner_identity.shape[2]),1)

        self.data = temp_
    
    def onehot(self,l,lon):
        temp_ = np.tile(np.array([False]),lon)
        temp_[np.array(l)] = True
        return temp_
    
    
    def setTrain(self):
        temp = self.data['person'].unique()
        self.test_id = np.array(self.data['person'].unique())[self.id___]
        self.train_id = np.delete(self.data['person'].unique(), self.id___)
        id_dict = {}
        for i in range(len(self.train_id)):
            id_dict[self.train_id[i]] = i
        for k in range(len(self.test_id)):
            id_dict[self.test_id[k]] = i+k+1
        self.data['person'] = self.data['person'].apply(lambda x :id_dict[x])
        self.train_id = list(id_dict.values())[:i+1]
        self.test_id = list(id_dict.values())[i+1:]
        label_dict = {}
        for i in range(len(self.data['label'].unique())):
            label_dict[self.data['label'].unique()[i]] = i
        self.data['label'] = self.data['label'].apply(lambda x :label_dict[x])
        self.data['outter_sum'] = self.data['outter'].apply(lambda x : np.mean(x),1)
        self.test_ind = self.select_id(self.test_id)
        self.train_ind = {}
        for id_ in self.train_id:
            temp_dict = {}
            for channel_ in self.data['channel'].unique():
                temp_dict[channel_] = self.certain_id(id_, channel_)
            self.train_ind[id_] = temp_dict
        self.reset()
        
    def reset(self):
        self.trainBatch = []
        if self.name == 'PAMAP2':
            self.train_ind = self.select_id(self.train_id)
            np.random.shuffle(self.train_ind)
            for i in range(len(self.train_ind)//256):
                self.trainBatch.append(self.train_ind[i*256:i*256+256])
            if i*256+256<len(self.train_ind):
                self.trainBatch.append(self.train_ind[i*256+256:])
        else:
            temp_ = []
            lon = 0
            for id_ in self.train_ind.keys():
                for key_ in self.train_ind[id_].keys():
                    np.random.shuffle(self.train_ind[id_][key_])
                    temp_ind = self.train_ind[id_][key_]
                    batch_ = []
                    for l_ in self.data['label'].unique():
                        temp_l_ = temp_ind[self.data.loc[temp_ind]['label']==l_]
                        num_ = (len(temp_l_)//self.batch+1)*self.batch
                        batch_.append(np.tile(temp_l_,2)[:num_].reshape(-1,self.batch))
                    lon_ = np.max([len(i) for i in batch_])
                    batch_ = [np.tile(i,(lon_//len(i)+1,1))[:lon_,:] for i in batch_]
                    temp_key = np.concatenate(batch_,1)
                    lon = np.max([temp_key.shape[0],lon])
                    temp_.append(temp_key)
            temp_ = [np.tile(i,(lon//i.shape[0]+1,1))[:lon,:] for i in temp_]
            self.trainBatch = np.concatenate(temp_, 1)
            self.trainBatch = [self.trainBatch[i,:] for i in range(self.trainBatch.shape[0])]
            
        
        
    def select_id(self, id_list):
        ind = [True if i in id_list else False for i in self.data['person'].values]
        return np.array(list(self.data.index))[ind]
    
    def certain_id(self, id_, channel_):
        return np.array(list(self.data[self.data['person']==id_][self.data['channel']==channel_].index))
    
    def getData(self, ind_list, test = False):
        difference_I = self.data['inner_up'][ind_list].values - self.data['inner_down'][ind_list].values
        I_up_loc = np.concatenate([i[np.newaxis,:] for i in self.data['inner_up_loc'][ind_list]],0)
        I_down_loc = np.concatenate([i[np.newaxis,:] for i in self.data['inner_down_loc'][ind_list]],0)
        #I_down_loc = self.down_loc(I_up_loc, self.data['inner'][0].shape[0])
        
        outter_pair_higher = []
        outter_pair_lower = []
        if test:
            if self.multi_channel:
                raw_data = np.concatenate([i[np.newaxis,:,:] for i in self.data['data'][ind_list].values],0)
            else:
                raw_data = np.concatenate(self.data['data'][ind_list].values,0).reshape(len(ind_list),-1)
            return np.concatenate([np.concatenate(self.data['inner'][ind_list].values,0).reshape(len(ind_list),-1), np.concatenate(self.data['outter'][ind_list].values,0).reshape(len(ind_list),-1)],1), raw_data, self.data['person'][ind_list].values, self.data['channel'][ind_list].values, self.data['label'][ind_list].values
        else:
            _ = self.data['outter_sum'][ind_list].argsort()
            ind_list = np.array(ind_list)[_]
            for i in range(len(ind_list)):
                higher = np.random.choice(np.arange(len(ind_list)//2,len(ind_list)),1)[0]
                outter_pair_higher.append(higher)
                outter_pair_lower.append(np.random.choice(np.arange(0,len(ind_list)//2),1)[0])
            higher = outter_pair_higher
            lower = outter_pair_lower
            outter_pair_higher = ind_list[outter_pair_higher]
            outter_pair_lower = ind_list[outter_pair_lower]
            difference_O = self.data['outter_sum'][outter_pair_higher].values - self.data['outter_sum'][outter_pair_lower].values
            
            """mark_multi_channel"""
            if self.multi_channel:
                raw_data = np.concatenate([i[np.newaxis,:,:] for i in self.data['data'][ind_list].values],0)
            else:
                raw_data = np.concatenate(self.data['data'][ind_list].values,0).reshape(len(ind_list),-1)
            return difference_I, I_up_loc, I_down_loc, difference_O, higher, lower, np.concatenate([np.concatenate(self.data['inner'][ind_list].values,0).reshape(len(ind_list),-1), np.concatenate(self.data['outter'][ind_list].values,0).reshape(len(ind_list),-1)],1), np.concatenate(self.data['frequency domain'][ind_list].values,0).reshape(len(ind_list),-1), raw_data, self.data['person'][ind_list].values, self.data['channel'][ind_list].values, self.data['label'][ind_list].values
            
    def down_loc(self, up_loc, shape):
        return [list(set(np.arange(shape))-set(i)) for i in up_loc]
    
    def getTrain(self):
        flag = True
        if len(self.trainBatch) > 0:
            train_ = self.trainBatch.pop(0)
        else:
            flag = False
            self.reset()
            train_ = self.trainBatch.pop(0)
        difference_I, I_up_loc, I_down_loc, difference_O, outter_pair_higher, outter_pair_lower, identity_weight, identity, raw_data, person, channel, label = self.getData(train_)
        return flag, difference_I, I_up_loc, I_down_loc, difference_O, outter_pair_higher, outter_pair_lower, identity_weight, identity, raw_data, person, channel, label
    
    def getTest(self):
        identity_weight, raw_data, person, channel, label = self.getData(self.test_ind, True)
        return identity_weight, raw_data, person, channel, label, person


class Motion(object):
    def __init__(self, fs, overlap, channel = 64, name = '', init = True, simple_flag = [], band_filter = False, multi_channel=False, people =9):
        self.channels = channel
        self.multi_channel = multi_channel
        self.people = people
        self.fs = fs
        self.overlap = overlap
        self.simple_flag = simple_flag
        self.band_filter = band_filter
        st = 'scratch/'
        if init and not os.path.isfile(st+name+"_all_raw.npy"):
            self.raw, self.data, self.label, self.person, self.inner_identity, self.outter_identity = self.read_data(name=name)
        else:
            self.raw = np.load(st+name+"_all_raw.npy")
            self.data = np.load(st+name+"_all_data.npy")
            self.inner_identity = np.load(st+name+"_inner_identity.npy")
            self.outter_identity = np.load(st+name+"_outter_identity.npy")
            self.label = np.load(st+name+"_all_label.npy")
            self.person = np.load(st+name+"_all_person.npy")
        print(self.label.shape)
            
        
        #self.data = (self.data-np.min(self.data))/(np.max(self.data)-np.min(self.data))
        self.data = F.tanh(torch.from_numpy(self.data)).data.cpu().numpy()
    
    def Fdomain(self, data, f_s = 160, N = 160, rep_ = 1, windowed = True, min_ = 0):        
        if windowed:
            window = signal.hann(len(data), sym = 0) * 2
            x10 = np.tile(data * window, rep_)
        else:
            x10 = np.tile(data, rep_)
        N *= rep_
        Fd = fftpack.fft(x10)
        f = fftpack.fftfreq(N, 1.0/f_s)
        masked = np.where(f > min_)
        mean_, sum_, std_ = np.mean(abs(Fd[masked])/N), np.sum(abs(Fd[masked])/N), np.std(abs(Fd[masked])/N)
        return Fd[:N//2+1], abs(Fd[masked])/N
    
       
    
    def read_data(self, name, normalize=False):
        all_data = []
        all_label = []
        all_person = []
        all_raw = []
        
        for i in range(1, self.people):
            print('file:{}'.format(str(i)))
            feature, label = load_one_subject(i, path='', dataset=name, window=self.fs,
                                          overlap=0.5, encoding=False, normalize=True,
                                          drop_redundant=False)
            if name == 'EEG':
                #4: image left hand, 5: image right hand
                fl = [True if i in [4,5] else False for i in label.squeeze()]
                ind_ = np.array([i for i in range(len(label))])[fl]
                feature = feature[ind_]
                label = label[ind_]
            all_label.append(label)
            all_person.append(np.tile([i],len(label)))
            all_data.append(feature)
        
        all_data = np.concatenate(all_data,0).squeeze()
        all_label = np.concatenate(all_label,0)
        all_person = np.concatenate(all_person,0)
        all_data = np.swapaxes(all_data,1,2)
        all_identity = []
        for dim_0 in range(all_data.shape[0]):
            print('dim:{}/{}'.format(str(dim_0),str(all_data.shape[0])))
            identity = []
            raw_ = []
            for dim_1 in range(all_data.shape[1]):
                raw, temp_ = self.Fdomain(all_data[dim_0,dim_1,:], f_s = self.fs, N = self.fs)
                identity.append(temp_)
                raw_.append(raw)
            all_identity.append(identity)
            all_raw.append(raw_)
        all_identity = np.array(all_identity)
        all_raw = np.array(all_raw)
        
        for dim_0 in range(all_identity.shape[1]):
            print('dim:{}/{}'.format(str(dim_0),str(all_identity.shape[1])))
            all_identity[:,dim_0,:] = pre.scale(all_identity[:,dim_0,:], 0)
        
        outter_identity = np.zeros(all_identity.shape)
        for dim_0 in range(all_identity.shape[1]):
            for dim_1 in range(all_identity.shape[2]):
                print('dim:{}/{},{}/{}'.format(str(dim_0),str(all_identity.shape[1]),str(dim_1),str(all_identity.shape[2])))
                temp_ = all_identity[:,dim_0,dim_1]
                outter_identity[:,dim_0,dim_1] += (temp_ - np.min(temp_)) / (np.max(temp_) - np.min(temp_))
        
        
        inner_identity = np.zeros(all_identity.shape)
        for dim_0 in range(all_identity.shape[0]):
            for dim_1 in range(all_identity.shape[1]):
                print('dim:{}/{},{}/{}'.format(str(dim_0),str(all_identity.shape[0]),str(dim_1),str(all_identity.shape[1])))
                temp_ = all_identity[dim_0,dim_1,:]
                inner_identity[dim_0,dim_1,:] += (temp_ - np.min(temp_)) / (np.max(temp_) - np.min(temp_))
        
        if self.band_filter:
            np.save(name+"_band_all_data.npy",all_data)
            np.save(name+"_band_all_raw.npy",all_raw)
            np.save(name+"_band_inner_identity.npy",inner_identity)
            np.save(name+"_band_outter_identity.npy",outter_identity)
        else:
            np.save('scratch/'+name+"_all_data.npy",all_data)
            np.save('scratch/'+name+"_all_raw.npy",all_raw)
            np.save('scratch/'+name+"_inner_identity.npy",inner_identity)
            np.save('scratch/'+name+"_outter_identity.npy",outter_identity)
            
        np.save('scratch/'+name+"_all_label.npy",all_label)
        np.save('scratch/'+name+"_all_person.npy",all_person)
        
        return all_raw, all_data, all_label, all_person, inner_identity, outter_identity
    
    def select_id(self, id_):
        return list(np.array([i for i in range(len(self.person))])[self.person==id_])
    
    def reshape(self, ratio = 0.2):
        self.num = int(self.inner_identity.shape[2] * ratio)
        self.in_sample = np.array([i for i in range(self.inner_identity.shape[2])])
        self.all_information = []
        for i in range(self.data.shape[1]):
            temp_ = pd.DataFrame(self.person,columns=['person'])
            temp_['channel'] = np.tile(i,self.data.shape[0])
            temp_['data'] = [self.data[j,i,:] for j in range(self.data.shape[0])]
            temp_['inner'] = [self.inner_identity[j,i,:] for j in range(self.data.shape[0])]
            temp_['outter'] = [self.outter_identity[j,i,:] for j in range(self.data.shape[0])]
            temp_['frequency domain'] = [self.raw[j,i,:] for j in range(self.raw.shape[0])]
            temp_['inner_up'] = pd.DataFrame(self.inner_identity[:,i,:]).apply(lambda x: np.mean(x[x.argsort()][-self.num:]),1)
            temp_['inner_down'] = pd.DataFrame(self.inner_identity[:,i,:]).apply(lambda x: np.mean(x[x.argsort()][:self.inner_identity.shape[2]-self.num]),1)
            temp_['inner_up_loc'] = pd.DataFrame(self.inner_identity[:,i,:]).apply(lambda x: list(self.in_sample[x.argsort()][-self.num:]),1)
            temp_['label'] = self.label
            self.all_information.append(temp_)
        self.all_information = pd.concat(self.all_information,0)
        self.all_information.index = range(len(self.all_information))
        
        

  
  

 
 
def main_process(i, select__ = [0, 0, 0], weight__ = False, gpu__ = 3, max_epoch_ = 50, seed=7, lr__ = [1e-4, 2e-4, 1e-5], align = False, fs=20, MCCNN_flag=False, simple_flag=[], lr_score_=1e-4,init_op=False):
         
    #写args，64channel    
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    
    #performance summary
    alllll_ = [] 
    person_alllll_ = [] 
    

    string_name_list = ['MHEALTH']
    step_list = [5]
    
    
    lr_score_ = lr_score_
    
    
    channel_list = [23, 36, 45, 6, 113, 15, 64]
    
    subject_list = [np.arange(10)[:,np.newaxis]]
    print('start')
    if MCCNN_flag:
        log_dir_name = 'MCCNN_' +  string_name_list[select__[0]] + '_'
    else:
        if weight__:
            log_dir_name = 'Weighted_' + string_name_list[select__[0]] + '_lr__' + str(lr__[select__[2]]) + '_step_' + str(step_list[select__[1]]) + 'score_lr' + str(lr_score_) + '_seed_' + str(seed)
        else:
            log_dir_name = 'Pur_' + string_name_list[select__[0]] + '_lr__' + str(lr__[select__[2]]) + '_step_' + str(step_list[select__[1]]) +  'score_lr' + str(lr_score_) + '_seed_' + str(seed)
        if align:
            log_dir_name += '_aligned'
    print('op')
    for subject_id in subject_list[select__[0]]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch", type=int, default=4)
        parser.add_argument("--max_epoch", type=int, default=max_epoch_)
        parser.add_argument("--fs", type=int, default=fs)
        parser.add_argument("--overlap", type=int, default=10)
        parser.add_argument("--init", type=bool, default=init_op)
        parser.add_argument("--simple_flag", type=list, default=simple_flag)
        parser.add_argument("--lr_score", type=float, default=lr_score_)
        parser.add_argument("--lr_encode", type=float, default=lr__[select__[2]])
        parser.add_argument("--lr_decode", type=float, default=lr__[select__[2]])
        parser.add_argument("--lr_dis", type=float, default=lr__[select__[2]])
        parser.add_argument("--weight_flag", type=bool, default=weight__)
        parser.add_argument("--log_dir", type=str, default=log_dir_name)
        parser.add_argument("--range", type=int, default=step_list[select__[1]])
        parser.add_argument("--band_filter",type=bool, default=False)
        parser.add_argument("--ratio",type=float, default=0.2)
        parser.add_argument("--multi_channel",type=bool, default=True)
        parser.add_argument("--gpu",type=int, default=gpu__)
        parser.add_argument("--modify",type=bool, default=False)
        parser.add_argument("--AE_frequency",type=bool, default=False)
        parser.add_argument("--channels",type=int, default=channel_list[select__[0]])
        parser.add_argument("--people",type=int, default=9)
        parser.add_argument("--id",type=int, default=list(subject_id))
        parser.add_argument("--align",type=bool, default=align)
        parser.add_argument("--name",type=str, default = string_name_list[select__[0]], help = 'name people dim fq MHEALTH 10 23, UCIDSADS 8 45, PAMAP2 8 36 100 USCHAD 14 6')
        op = parser.parse_args()
        data = Motion(fs = op.fs, overlap = op.overlap, channel = op.channels, name = op.name, init = op.init, simple_flag = op.simple_flag, band_filter = op.band_filter, multi_channel=op.multi_channel, people = op.people)
        print(data.label.shape)
        max_epoch = op.max_epoch            
        
        if not op.multi_channel:
            data.reshape(ratio=op.ratio)
            exp = experiment(data.all_information, op.batch, op.multi_channel, op.ratio, op.name)
        else:
            print('op')
            exp = experiment(data, op.batch, op.multi_channel, op.ratio, op.name, op.id)
            print('op')
        gpu = 'cuda:'+str(op.gpu) if torch.cuda.is_available() else 'cpu'
        #net initialization
        #score, encoder, decoder, discriminator = Score(op.fs*10-2,op.fs*5-1), Encoder(op.fs, max(10,op.fs//2), max(10,op.fs//4), op.multi_channel, op.channels), Decoder(max(10,op.fs//4), max(10,op.fs//2), op.fs, op.multi_channel, op.channels), Discriminator(op.fs, max(10,op.fs//2), op.channels, len(set(data.label.squeeze())), len(exp.train_id), op.multi_channel)
        #score, iencoder, nencoder, decoder, discriminator = Score(op.fs*10-2,op.fs*5-1), MCCNN(op.fs, op.channels), MCCNN(op.fs, op.channels), deMCCNN(op.fs, op.channels), DD(op.channels * 20, op.channels * 10, op.channels, len(set(data.label.squeeze())), len(exp.train_id))


        if MCCNN_flag:
            score, encoder, decoder, discriminator = Score(op.fs*10-2,op.fs*5-1), MCCNN(op.fs, op.channels), deMCCNN(op.fs, op.channels), pMCCNN(op.fs, op.channels, len(set(data.label.squeeze())), len(exp.train_id))
        
            #score.to(gpu)
            #encoder.to(gpu)
            #decoder.to(gpu)
            discriminator.to(gpu)
            optim_ = torch.optim.Adam([#{'params':score.parameters(),'lr':op.lr_score},
                                       #{'params':encoder.parameters(),'lr':op.lr_encode},
                                       #{'params':decoder.parameters(),'lr':op.lr_decode},
                                       {'params':discriminator.parameters(),'lr':1e-4}
                                      ])
        else:
            print('model')
            iencoder, nencoder, decoder, discriminator = MCCNN(op.fs, op.channels), MCCNN(op.fs, op.channels), deMCCNN(op.fs, op.channels), DD(op.channels * 20, op.channels * 10, op.channels, len(set(data.label.squeeze())), len(exp.train_id))
            print('model')
            
            iencoder.to(gpu)
            nencoder.to(gpu)
            decoder.to(gpu)
            discriminator.to(gpu)
            
            optim_ = torch.optim.Adam([{'params':iencoder.parameters(),'lr':op.lr_encode},
                                       {'params':decoder.parameters(),'lr':op.lr_decode},
                                       {'params':discriminator.parameters(),'lr':op.lr_dis}
                                      ])
            optim_2 = torch.optim.Adam([{'params':nencoder.parameters(),'lr':op.lr_encode/10}])
            if op.weight_flag:
                score = Score(op.fs-2,int(op.fs*0.5)-1)
                score.to(gpu)
                optim_score = torch.optim.Adam([{'params':score.parameters(),'lr':op.lr_score}])
                score.apply(weights_init_normal)
        
        
        #train data
        
        
        A_set = []
        label_unique_element = list(exp.data['label'].unique())
        A_set_person = [[] for __ in op.id]
        epoch = 0
        iteration = 0
        writer = SummaryWriter()
        weight_flag = op.weight_flag
        test_identity_weight, test_raw_data, test_person, test_channel, test_label, test_person = exp.getTest()
        test_label = np.array(test_label).squeeze()
        test_person = np.array(test_person).squeeze()
        test_raw_data = torch.from_numpy(test_raw_data).float()
        test_identity_weight = torch.from_numpy(test_identity_weight).float()
        max_ = 0
        print('start')


        epoch = 0
        iteration = 0
        max_acc_ = 0
        best_model_ = []
        while epoch < max_epoch:
            iteration += 1
            flag, difference_I, I_up_loc, I_down_loc, difference_O, outter_pair_higher, outter_pair_lower, identity_weight, identity, raw_data, person, channel,label = exp.getTrain()
            if not flag:
                epoch += 1
            raw_data = torch.from_numpy(raw_data).float().to(gpu)
            label = torch.from_numpy(label).long().to(gpu)
            if op.weight_flag:
                difference_O = (difference_O - np.min(difference_O) - 1e-5) / (
                                np.max(difference_O) - np.min(difference_O))
                difference_I = (difference_I - np.min(difference_I) - 1e-5) / (
                                np.max(difference_I) - np.min(difference_I))
                difference_I = torch.from_numpy(difference_I).float().to(gpu)
                difference_O = torch.from_numpy(difference_O).float().to(gpu)
                identity_weight = torch.from_numpy(identity_weight.astype('float')).float().to(gpu)
                cri = torch.tensor(-1.0).float().to(gpu)
                optim_score.zero_grad()
                score_ = score(identity_weight)
                up_i = torch.mean(score_[torch.from_numpy(I_up_loc)].reshape(score_.shape[0], -1), 1)
                down_i = torch.mean(score_[torch.from_numpy(I_down_loc)].reshape(score_.shape[0], -1), 1)
                count_i = up_i - down_i
                loss_i = 100 * torch.mean(
                        torch.abs(difference_I - count_i) * count_i * (cri ** (difference_I > count_i).float()))
                higher_score = torch.mean(score_[outter_pair_higher, :], 1)
                lower_score = torch.mean(score_[outter_pair_lower, :], 1)
                count_o = higher_score - lower_score
                loss_o = 100 * torch.mean(
                        torch.abs(difference_O - count_o) * count_o * (cri ** (difference_O > count_o).float()))
                score_loss = loss_i + loss_o
                score_loss.backward()
                optim_score.step()
                criterion_score = torch.mean(torch.abs(difference_I - count_i)) + torch.mean(
                        torch.abs(difference_O - count_o))
                if iteration % 20 == 0:
                    writer.add_scalar(op.log_dir + '/runs/Loss/criterion', criterion_score.data.cpu().numpy(),
                                          iteration)
                    writer.add_scalar(op.log_dir + '/runs/Loss/inner', loss_i.data.cpu().numpy(), iteration)
                    writer.add_scalar(op.log_dir + '/runs/Loss/outter', loss_o.data.cpu().numpy(), iteration)

                weight = torch.mean(score_, 1).detach()
                class_weight = torch.zeros(len(exp.data['label'].unique())).float().to(gpu)
                score.eval()
                for class_label in exp.data['label'].unique():
                    class_weight[class_label] += torch.mean(weight[label == class_label])
                task_loss_cls = torch.nn.CrossEntropyLoss(weight=class_weight).to(gpu)
                # task_loss_cls = torch.nn.CrossEntropyLoss(reduce=False).to(gpu)

            else:
                task_loss_cls = torch.nn.CrossEntropyLoss(reduce=False).to(gpu)
            #MCCNN
            if MCCNN_flag:
                #print('MCCNN')
                discriminator.train()
                information_label = discriminator(raw_data)
                task_loss_cls = torch.nn.CrossEntropyLoss().to(gpu)
                loss = task_loss_cls(information_label,label)
                loss.backward()
                optim_.step()
                if epoch > 0 and iteration % 20 == 0:
                    #test data
                    #encoder.eval()
                    #decoder.eval()
                    discriminator.eval()
                    res = []
                    for test_num in range(test_raw_data.shape[0]//64):
                        temp_test = test_raw_data[test_num*64:test_num*64+64,:,:].to(gpu)
                        information_label = discriminator(temp_test)
                        information_label = information_label.detach()
                        res.append(torch.argmax(information_label,1).data.cpu().numpy())
                    if test_num*64+64<test_raw_data.shape[0]:
                        temp_test = test_raw_data[test_num*64+64:,:,:].to(gpu)
                        information_label = discriminator(temp_test)
                        information_label = information_label.detach()
                        res.append(torch.argmax(information_label,1).data.cpu().numpy())
                    res = np.concatenate(res,0).squeeze()
                    acc=(np.array(res)==np.array(test_label)).sum()/len(test_label)
                    A_set = save_set(iteration,classification_report(np.array(test_label),np.array(res),output_dict=True),A_set,len(set(test_label)))
                    A_set[-1][1].append(acc)
                    if len(op.id)>1:
                        for temp_user in range(len(op.id)):
                            temp_user_id = exp.test_id[temp_user]
                            temp_test_label = np.array(test_label)[test_person==temp_user_id]
                            temp_res = np.array(res)[test_person==temp_user_id]
                            A_set_person[temp_user] = save_set(iteration,classification_report(temp_test_label, temp_res, output_dict=True),A_set_person[temp_user],len(set(test_label)))
                            temp_acc = (temp_test_label==temp_res).sum()/len(temp_test_label)
                            A_set_person[temp_user][-1][1].append(temp_acc)
                    print(classification_report(test_label,res))
                    print('test_acc:{}'.format(acc))
                    
            else:
                print(iteration)
                for ___ in range(op.range):
                    if op.weight_flag:
                        score_ = score(identity_weight)
                        weight = torch.mean(score_, 1).detach()
                    else:
                        weight = torch.ones(identity_weight.shape[0]).float().to(gpu)
                    iencoder.train()
                    nencoder.eval()
                    discriminator.train()
                    decoder.train()
                    optim_.zero_grad()
                    iembedding = iencoder(raw_data)
                    nembedding = nencoder(raw_data).detach()
                    generated = decoder(iembedding + nembedding)
                    #AE loss
                    AE_loss = torch.mean(weight * torch.mean(torch.mean((generated - raw_data)**2,1),1))

                    
                    information_label = discriminator(iembedding)
                    #information_loss = task_loss_person(information_domain, person) + 10 * task_loss_cls(information_label,label)
                    #information_loss = torch.mean(weight*task_loss_cls(information_label,label))
                    information_loss = task_loss_cls(information_label, label)
                        
                    
                    train_acc = (torch.argmax(information_label,1).data.cpu().numpy()==label.data.cpu().numpy()).sum()/len(label)
                    discriminator_loss = information_loss
                    #print('information loss: {}'.format(information_loss.data.cpu().numpy()))
                    
                                            
                    
                    #print('align loss: {}'.format(align_loss.data.cpu().numpy()))
                    #all_loss = score_loss + discriminator_loss + AE_loss + align_loss
                    all_loss = discriminator_loss + AE_loss
                    
                    #alignment
                    if op.align:
                        align_loss = torch.tensor(0.0).float().to(gpu)
                        for each in exp.train_id:
                            align_loss += torch.mean(weight[person==each] * torch.mean(torch.abs(iembedding[person==each] - torch.mean(iembedding[person==each],0)),1))
                        for each in range(len(exp.data['channel'].unique())):
                            align_loss += torch.mean(weight[channel==each] * torch.mean(torch.abs(iembedding[channel==each] - torch.mean(iembedding[channel==each],0)),1))
                        all_loss += align_loss
                    """
                    print(weight.shape)
                    print(iembedding.shape)
                    print(label.shape)
                    print(weight[label==each].shape)
                    print(iembedding[label==each].shape)
                    print(torch.mean(iembedding[label==each],0).shape)
                    """
                        
 
                    #print('all loss: {} \n'.format(all_loss.data.cpu().numpy()))
                    all_loss.backward()
                    optim_.step()
                    
                writer.add_scalar(op.log_dir+'/runs/Accuracy/train', train_acc ,iteration)
                iencoder.eval()
                nencoder.train()
                discriminator.eval()
                decoder.eval()
                optim_2.zero_grad()
                nembedding = nencoder(raw_data)    
                noise_label = discriminator(nembedding)
                if op.weight_flag:
                    score_ = score(identity_weight)
                    weight = torch.mean(score_, 1).detach()
                    noise_loss = - task_loss_cls(noise_label, label)
                else:
                    weight = torch.ones(identity_weight.shape[0]).float().to(gpu)
                    noise_loss = - torch.mean(weight*task_loss_cls(noise_label,label))
                noise_loss.backward()
                optim_2.step()
                
                
                if op.align and iteration % 20 == 0:
                    writer.add_scalar(op.log_dir+'/runs/Loss/criterion', align_loss.data.cpu().numpy() ,iteration)
                if iteration % 20 == 0:
                    writer.add_scalar(op.log_dir+'/runs/Loss/AE', AE_loss.data.cpu().numpy() ,iteration)
                    writer.add_scalar(op.log_dir+'/runs/Loss/noise_loss', noise_loss.data.cpu().numpy() ,iteration)
                    writer.add_scalar(op.log_dir+'/runs/Loss/information_loss', information_loss.data.cpu().numpy() ,iteration)
                if epoch > -1 and iteration % 20 == 0:
                    #test data
                    iencoder.eval()
                    nencoder.eval()
                    decoder.eval()
                    discriminator.eval()
                    res = []
                    for test_num in range(test_raw_data.shape[0]//64):
                        temp_test = test_raw_data[test_num*64:test_num*64+64,:,:].to(gpu)
                        embedding = iencoder(temp_test).detach()
                        information_label = discriminator(embedding.detach())
                        information_label = information_label.detach()
                        res.append(torch.argmax(information_label,1).data.cpu().numpy())
                    if test_num*64+64<test_raw_data.shape[0]:
                        temp_test = test_raw_data[test_num*64+64:,:,:].to(gpu)
                        embedding = iencoder(temp_test).detach()
                        information_label = discriminator(embedding.detach())
                        information_label = information_label.detach()
                        res.append(torch.argmax(information_label,1).data.cpu().numpy())
                    res = np.concatenate(res,0).squeeze()
                    acc=(np.array(res)==np.array(test_label)).sum()/len(test_label)
                    if acc > max_acc_:
                        max_acc_ = acc
                        best_model_ = []
                        best_model_.append(iencoder.state_dict())
                        best_model_.append(discriminator.state_dict())
                    #print(classification_report(test_label,res))
                    print('test_acc:{}'.format(acc))
                    A_set = save_set(iteration,classification_report(np.array(test_label),np.array(res),output_dict=True),A_set,len(set(test_label)))
                    A_set[-1][1].append(acc)
                    if len(op.id)>1:
                        for temp_user in range(len(op.id)):
                            temp_user_id = exp.test_id[temp_user]
                            temp_test_label = np.array(test_label)[test_person==temp_user_id]
                            temp_res = np.array(res)[test_person==temp_user_id]
                            A_set_person[temp_user] = save_set(iteration,classification_report(temp_test_label, temp_res, output_dict=True),A_set_person[temp_user],len(set(test_label)))
                            temp_acc = (temp_test_label==temp_res).sum()/len(temp_test_label)
                            A_set_person[temp_user][-1][1].append(temp_acc)
                    
                    writer.add_scalar(op.log_dir+'/runs/Accuracy/test', acc ,iteration)
                    iencoder.train()
                    nencoder.train()
                    decoder.train()
                    discriminator.train()
                    max_ = max(max_,acc)
        print('model/iencoder_'+op.name+'_'+str(subject_id[0])+'.pt')
        torch.save(best_model_[0],'model/iencoder_'+op.name+'_'+str(subject_id[0])+'.pt')
        torch.save(best_model_[1],'model/discriminator_'+op.name+'_'+str(subject_id[0])+'.pt')

        print(max_)
        writer.close()
        columns = []
        #for i in range(len(set(exp.data['label'].unique()))):
        #    columns.append(str(i)+'-precision')
        #    columns.append(str(i)+'-recall')
        #    columns.append(str(i)+'-f1-score')
        columns.extend(['micro avg-precision','micro avg-recall','micro avg-f1-score','macro avg-precision','macro avg-recall','macro avg-f1-score','weighted avg-precision','weighted avg-recall','weighted avg-f1-score','acc'])
        A_set = build(columns,A_set)
        
        temp_set = A_set.sort_values('acc')[-1:]
        if len(op.id)>1:
            A_set_person = [build(columns,_) for _ in A_set_person]
            temp_set_person = [_.loc[temp_set.index].values[0] for _ in A_set_person]
            for _ in range(len(temp_set_person)):
                print('person:{}'.format(str(op.id[_])))
                print(temp_set_person[_])
                person_alllll_.append([_,temp_set_person[_]])
                
        print(temp_set)
        print(op.id)
        
        
        #l=list(temp_set.values)
        #l.append(op.id)
        alllll_.append([op.id[0],temp_set.values[0]])
        
    need_to_save = build(columns,alllll_)
    if len(op.id)>1:
        need_to_save_person = build(columns,person_alllll_)
        need_to_save_person.to_csv('res/Person'+op.log_dir+'.csv',index=False)
    else:
        need_to_save.to_csv('res/'+op.log_dir+'.csv',index=False)
    

        
import torch.multiprocessing.spawn as sp 
import time

if __name__=='__main__':
    #dataset
    name = 'MHEALTH'
    #spectrum controller
    weight__ = True
    #SOTA-MCCNN controller
    MCCNN_flag = False
    #gpu
    gpu__list = [0]
    
    fs = 20
    seed = 50
    step = 5
    max_epoch = 30 
    lr_score_ = 1e-4
    
    lr__ = [2e-4]
    
    p = []
    
    #align data-unused
    align = False
    #choose subset-unnused
    simple_flag = []
    max_epoch = 50
    
    #pool
    p = []
    #10 subjects in MHEALTH
    order_list = [[0,0,i] for i in range(10)]
    
    #initialize data for first time
    init_op = True
    
    for i in np.arange(len(order_list)):
        gpu__ = gpu__list[i%len(gpu__list)]
        p.append(sp(main_process, args=(order_list[i], weight__, gpu__, max_epoch, seed, lr__, align, fs, MCCNN_flag,simple_flag, lr_score_,init_op), join=False))
        time.sleep(50)
        
            
    for _ in p:
        _.join()        