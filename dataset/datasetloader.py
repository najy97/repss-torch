import h5py
import numpy as np
import os
from dataset.CustomDataset import Custom_Dataset

def datasetloader(save_root_path: str = "C:/preprocessed_dataset/",
                  dataset_root_path: str = "C:/VIPL_HR/data",
                  option: str = "train"):
    video_data = []
    label_data = []
    hpy_file = h5py.File(save_root_path+option+'.hdf5',"r")
    for key in hpy_file.keys():
