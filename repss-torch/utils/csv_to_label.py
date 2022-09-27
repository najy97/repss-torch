import numpy as np
import h5py
import scipy.signal
import pandas as pd


def label_preprocessor(path, length):
    f = pd.read_csv(path)
    label = f['Wave']
    label = np.array(label).astype('float32')
    label = scipy.signal.resample(label, length)

    split_raw_label = np.zeros(((len(label)//32), 32))
    index = 0
    for i in range(len(label)//32):
        split_raw_label[i] = label[index:index+32]
        index += 32
    return split_raw_label
