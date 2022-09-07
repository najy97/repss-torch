import h5py
import numpy as np
import os
import random
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class Custom_Dataset(Dataset):
    def __init__(self, video_data, label_data):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = video_data
        self.label = label_data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # video data 축 바꾸기 (length, width, height, channel) -> (channel, length, width, height)
        # conv3d의 input shape (batch, channel, depth, width, height)
        video_data = torch.tensor(np.transpose(self.video_data[index], (3, 0, 1, 2)), dtype=torch.float32)
        label_data = torch.tensor(self.label[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video_data = video_data.to('cuda')
            label_data = label_data.to('cuda')

        return video_data, label_data

    def __len__(self):
        return len(self.label)
