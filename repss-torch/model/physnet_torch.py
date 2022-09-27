# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# custom blocks
from model.DecoderBlock import decoder_block
from model.EncoderBlock import encoder_block

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import time
import copy


class PhysNet(nn.Module):
    def __init__(self, frames=32):
        super(PhysNet, self).__init__()
        self.physnet = nn.Sequential(
            encoder_block(),
            decoder_block(),
            nn.AdaptiveMaxPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        return self.physnet(x).view(-1, length)
