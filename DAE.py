# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import math
import itertools
from scipy import fftpack,signal

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import classification_report

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import conv1d

class deConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deConvBlock, self).__init__()
        conv_block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(5, 1), padding=0),
                      nn.ReLU(inplace=True),
                      #nn.MaxPool2d(kernel_size=(2, 1)),
                      nn.BatchNorm2d(num_features=out_channels)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        #shape of x is [B, C, W, H]
        return self.conv_block(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        conv_block = [nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=0),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=(2, 1)),
                      nn.BatchNorm2d(num_features=out_channels)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        #shape of x is [B, C, W, H]
        return self.conv_block(x)

class MCCNN(nn.Module):
    def __init__(self, window_size = 20, sensor_channels = 23):
        super(MCCNN, self).__init__()
        self.window_size = window_size
        self.sensor_channels = sensor_channels       #23 for MHEALTH, 36 for PAMAP2
        self.conv1 = ConvBlock(1, 50)
        self.conv2 = ConvBlock(50, 40)
        self.conv3 = nn.Sequential(
                        nn.Conv2d(40, 20, kernel_size=(((window_size-4)//2-4)//2, 1), padding=0),
                        nn.ReLU(inplace=True),
                        #nn.MaxPool2d(kernel_size=(2, 1)),
                        nn.BatchNorm2d(20))

    def forward(self, x):
        #shape of x is [B, time_window, dim]
        #reshape to [B, channel, time_window, dim]
        x = x.transpose(1,2)
        if x.dim() < 3 or x.dim() > 4:
            raise Exception("input dim error")
        elif x.dim() == 3:
            #x.unsqueeze(1)
            x = torch.unsqueeze(x, dim=1)
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        return out

class deMCCNN(nn.Module):
    def __init__(self, window_size = 20, sensor_channels = 23):
        super(deMCCNN, self).__init__()
        self.window_size = window_size
        self.sensor_channels = sensor_channels       #23 for MHEALTH, 36 for PAMAP2
        self.conv1 = deConvBlock(20, 40)
        self.conv2 = deConvBlock(40, 50)
        self.conv3 = nn.Sequential(
                        nn.ConvTranspose2d(50, window_size//10, kernel_size=(2, 1), padding=0),
                        nn.ReLU(inplace=True),
                        #nn.MaxPool2d(kernel_size=(2, 1)),
                        nn.BatchNorm2d(window_size//10)
                        )
        """
        self.conv4 = nn.Sequential(
                        nn.ConvTranspose2d(50, window_size//10, kernel_size=(2, 1), padding=0),
                        nn.ReLU(inplace=True),
                        #nn.MaxPool2d(kernel_size=(2, 1)),
                        nn.BatchNorm2d(window_size//10)
                        )
        """

    def forward(self, x):
        #shape of x is [B, time_window, dim]
        #reshape to [B, channel, time_window, dim]
        x = x.reshape(x.size(0), 20, 1, self.sensor_channels)
        
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        noise = self.conv3(out).reshape(out.size(0), self.sensor_channels, self.window_size)
        #print(out.shape)
        #pur = self.conv4(out).reshape(out.size(0), self.sensor_channels, self.window_size)
        #print(out.shape)
        #out = self.fc(out)
        #out = self.softmax(out)
        #return noise, pur
        return noise
    
class pMCCNN(nn.Module):
    def __init__(self, window_size, channels, labels, domains):
        super(pMCCNN, self).__init__()
        self.window_size = window_size
        self.sensor_channels = channels       #23 for MHEALTH, 36 for PAMAP2
        self.conv1 = ConvBlock(1, 50)
        self.conv2 = ConvBlock(50, 40)
        self.conv3 = nn.Sequential(
                        nn.Conv2d(40, 20, kernel_size=(((window_size-4)//2-4)//2, 1), padding=0),
                        nn.ReLU(inplace=True),
                        #nn.MaxPool2d(kernel_size=(2, 1)),
                        nn.BatchNorm2d(20))
        """
        self.channel = nn.Sequential(
                nn.Linear(20*self.sensor_channels, channels),
                nn.Softmax(dim=1)
                )
        
        self.domain = nn.Sequential(
                nn.Linear(20*self.sensor_channels, domains),
                nn.Softmax(dim=1)
                )
        """
        self.label = nn.Sequential(
                nn.Linear(20*self.sensor_channels, labels),
                nn.Softmax(dim=1)
                )
        
    def forward(self, x):
        #shape of x is [B, time_window, dim]
        #reshape to [B, channel, time_window, dim]
        
        x = x.transpose(1,2)
        if x.dim() < 3 or x.dim() > 4:
            raise Exception("input dim error")
        elif x.dim() == 3:
            #x.unsqueeze(1)
            x = torch.unsqueeze(x, dim=1)
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        #return self.channel(out), self.label(out), self.domain(out)
        return self.label(out)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.xavier_normal(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class Score(nn.Module):
    def __init__(self, in_features, out_features):
        super(Score, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, data):
        score = self.model(data)
        #score = (score-torch.min(score))/(torch.max(score)-torch.min(score))
        return score
    
class Encoder(nn.Module):
    def __init__(self, in_features, hidden, out_features, flag, channels=64):
        super(Encoder, self).__init__()
        self.flag = flag
        self.channels = channels
        if self.flag:
            self.attention = nn.Conv1d(self.channels, 1, 1)
        self.model = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_features),
                nn.Tanh()
                )
    def forward(self, data):
        if self.flag:
            data = self.attention(data).squeeze()
        representation = self.model(data)
        return representation

class Decoder(nn.Module):
    def __init__(self, in_features, hidden, out_features, flag, channels=64):
        super(Decoder, self).__init__()
        self.flag = flag
        self.channels = channels
        if flag:
            self.information = nn.Sequential(
                    nn.Linear(in_features, hidden * self.channels),
                    nn.Tanh(),
                    nn.Linear(hidden * self.channels, out_features * self.channels)
                    )
            
            self.noise = nn.Sequential(
                    nn.Linear(in_features, hidden * self.channels),
                    nn.Tanh(),
                    nn.Linear(hidden * self.channels, out_features * self.channels)
                    )
        else:
            
            self.information = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, out_features)
                    )
            
            self.noise = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, out_features)
                    )
        
    def forward(self, data):
        noise_ = self.noise(data)
        information_ = self.information(data)
        if self.flag:
            noise_ = noise_.reshape(data.shape[0],self.channels,-1)
            information_ = information_.reshape(data.shape[0],self.channels,-1)
        return noise_, information_
    
#data:{batch_size, lon}
def win_fft(data, time = 10, gpu = 'cpu', multi_channel=False):
    if multi_channel:
        window = torch.from_numpy(signal.hann(data.shape[2], sym = 0) * 2).float().to(gpu)
        temp_data = data * window
        return torch.mean(torch.rfft(temp_data.repeat((1,1,time)),2,normalized=True)[:,:,1:(time*data.shape[2])//2,:],1)
    else:
        window = torch.from_numpy(signal.hann(data.shape[1], sym = 0) * 2).float().to(gpu)
        temp_data = data * window
        return torch.rfft(temp_data.repeat((1,time)),1,normalized=True)[:,1:(time*data.shape[1])//2,:]

class DD(nn.Module):
    def __init__(self, in_features, hidden, channels, labels, domains):
        super(DD, self).__init__()
        
        
        self.label = nn.Sequential(
                #nn.Linear(in_features, hidden),
                #nn.Tanh(),
                #nn.Linear(hidden, labels),
                nn.Linear(in_features, labels),
                nn.Softmax(1)
                )
    def forward(self, data):
        return self.label(data)
    
    
class Discriminator(nn.Module):
    def __init__(self, in_features, hidden, channels, labels, domains, flag):
        super(Discriminator, self).__init__()
        self.flag = flag
        self.channels = channels
        if self.flag:
            self.attention = nn.Conv1d(self.channels, 1, 1)
        self.embed = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh()
                )
        self.channel = nn.Sequential(
                nn.Linear(hidden, channels),
                nn.Softmax(1)
                )
        
        self.label = nn.Sequential(
                nn.Linear(hidden, labels),
                nn.Softmax(1)
                )
        self.domain = nn.Sequential(
                nn.Linear(hidden, domains),
                nn.Softmax(1)
                )
        
    def forward(self, data):
        if self.flag:
            data = self.attention(data).squeeze()
        self.embedding = self.embed(data)
        return self.channel(self.embedding), self.label(self.embedding), self.domain(self.embedding)