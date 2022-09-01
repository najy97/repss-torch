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


def Video_Sampler(path):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FPS))


    fps = 30
    t = 128
    count = 0
    video = []

    while cap.isOpened():
        ret, img = cap.read()

        if int(cap.get(1)) % fps == 0:
            count += 1
            video.append(img)
            print(count)
        if count >= t:
            break

    return video


video = Video_Sampler('C:/V4V/train_val/data/F001_T1/video.mkv')
print(np.shape(video))
