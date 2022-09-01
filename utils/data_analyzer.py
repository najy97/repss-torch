import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict

list_name = defaultdict(list)
list_seconds = defaultdict(list)
list_frames = defaultdict(list)

src_list = ['source1', 'source2', 'source3', 'source4']

for (path, dir, files) in os.walk('C:/VIPL_HR/data'):
    for filename in files:
        for src in src_list:
            if '.avi' in filename and src in path:
                cap = cv2.VideoCapture(path + '/' + filename)
                name = path[16:].replace('\\', '')
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                seconds = int(frames / fps)

                list_name[src].append(name)
                list_seconds[src].append(seconds)
                list_frames[src].append(frames)

for src in src_list:
    df_name = pd.DataFrame(list_name[src]).transpose()
    df_seconds = pd.DataFrame(list_seconds[src]).transpose()
    df_frames = pd.DataFrame(list_frames[src]).transpose()

    avg_seconds = np.mean(list_seconds[src])
    avg_frames = np.mean(list_frames[src])

    df = pd.concat([df_name, df_seconds, df_frames])
    # df.to_csv(src+'.csv', index=False, mode='w', encoding='utf-8')
    print(df)
