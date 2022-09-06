import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm


def video_preprocessor(path):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_video = np.empty((length, 128, 128, 3))
    j = 0

    with tqdm(total=length, position=0, leave=True, desc=path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break

            crop_frame = cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            crop_frame = cv2.cvtColor(crop_frame.astype('float32'), cv2.COLOR_BGR2RGB)
            crop_frame[crop_frame > 1] = 1
            crop_frame[crop_frame < 1e-6] = 1e-6

            raw_video[j] = crop_frame
            j += 1
            pbar.update(1)
        cap.release()

    split_raw_video = np.zeros(((length // 32), 32, 128, 128, 3))
    index = 0
    for x in range(length//32):
        split_raw_video[x] = raw_video[index:index+32]
        index += 32

    return split_raw_video
