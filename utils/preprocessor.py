# import package

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models


from torchvision import utils
import matplotlib.pyplot as plt

import numpy as np
from torchsummary import summary
import time
import copy
import cv2

from utils.video_to_dataset import video_preprocessor
from utils.csv_to_label import label_preprocessor
import multiprocessing
import h5py
import os

def preprocessing():
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

def Preprocess_Dataset(path, return_dict):
    for v in os.listdir(path):
        for source in os.listdir(path+'/'+v):
            if source != 'source4':
                preprocessed_video = video_preprocessor(path+'/'+v+'/'+source+'/video.avi')
                preprocessed_label = label_preprocessor(path + '/' + v + '/' + source + '/wave.csv',
                                                        preprocessed_video.shape[0]*preprocessed_video.shape[1])  # n 블록 * 32 프레임
                return_dict[path[-3:]+'_'+v+'_'+source] = {'preprocessed_video': preprocessed_video,
                                                           'preprocessed_label': preprocessed_label}

if __name__=='__main__':
