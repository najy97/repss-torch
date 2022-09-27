import h5py
import numpy as np
import os
from dataset.CustomDataset import Custom_Dataset


def datasetloader(save_root_path: str = "C:/preprocessed_dataset/",
                  option: str = "train"):
    video_data = []
    label_data = []
    hpy_file = h5py.File(save_root_path + option + '.hdf5', "r")
    for key in hpy_file.keys():
        video_data.extend(hpy_file[key]['preprocessed_video'])
        label_data.extend(hpy_file[key]['preprocessed_label'])
        hpy_file.close()

    dataset = Custom_Dataset(video_data=np.asarray(video_data),
                             label_data=np.asarray(label_data))
    return dataset
