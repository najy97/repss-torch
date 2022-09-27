from utils.video_to_dataset import video_preprocessor
import os
import sys

path = '/VIPL_HR/data/p20/v1/source1/video.avi'
print('video path :', path)
print('video original size :', os.path.getsize(path) // 1048576, 'Mb')
preprocessed = video_preprocessor(path)
print('preprocessed video size :', sys.getsizeof(preprocessed) // 1048576, 'Mb')
