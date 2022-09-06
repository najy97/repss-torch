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
import os

from torchvision import utils
import matplotlib.pyplot as plt

import numpy as np
from torchsummary import summary
import time
import copy
import cv2


def Video_Sampler(path, fps=25):
    cap = cv2.VideoCapture(path)
    cap = cap.set(cv2.CAP_PROP_POS_MSEC, fps)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    while cap.isOpened():
        # file name setting
        file_name = path.split('/')
        file_name = ''.join(file_name[-3:])

        # get image from video
        ret, img = cap.read()

        out = cv2.VideoWriter(file_name+str(count)+'.avi', fourcc, 30.0, 128, 128)

        if not ret:
            break





    return video


video = Video_Sampler('C:/V4V/train_val/data/F001_T1/video.mkv')
print(np.shape(video))
