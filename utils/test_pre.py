# import package

from video_to_dataset import video_preprocessor
from csv_to_label import label_preprocessor
import multiprocessing
import h5py
import os
from tqdm import tqdm


def preprocessing(save_root_path: str = "/preprocessed_dataset/",
                  dataset_root_path: str = "/VIPL_HR/data",
                  train_ratio: float = 0.8,
                  split_index: list = [0, 0],
                  index_train_name: int = 0,
                  index_test_name: int = 0):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    data_list = [data for data in os.listdir(dataset_root_path)]  # p1 - p107

    process = []

    # multiprocessing
    for i in range(split_index[0], split_index[1]+1):
        proc = multiprocessing.Process(target=Preprocess_Dataset,
                                       args=(dataset_root_path + '/' + data_list[i], return_dict))
        process.append(proc)  # join할 프로세스 리스트 생성
        proc.start()  # 프로세스 시작 신호

    for proc in process:
        proc.join()  # 해당 프로세스가 종료될 때 까지 대기

    train = int(len(return_dict.keys()) * train_ratio)  # train/validation 분할

    pbar = tqdm(total=train, position=0, leave=True, desc='Saving Train Dataset')
    train_file = h5py.File(save_root_path + 'train_' + str(index_train_name) + '.hdf5', "w")
    for index, data_path in enumerate(return_dict.keys()[:train]):
        # hdf5파일 생성(열기) 및 쓰기
        index_train_name += 1
        dset = train_file.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        pbar.update(1)
    train_file.close()
    pbar.close()

    pbar = tqdm(total=(len(return_dict.keys())-train), position=0, leave=True, desc='Saving Test Dataset')
    test_file = h5py.File(save_root_path + "test_" + str(index_test_name) + '.hdf5', "w")
    for index, data_path in enumerate(return_dict.keys()[train:]):
        index_test_name += 1
        dset = test_file.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        pbar.update(1)
    test_file.close()
    pbar.close()

    return index_train_name, index_test_name


def Preprocess_Dataset(path, return_dict):
    for v in os.listdir(path):
        for source in os.listdir(path + '/' + v):
            if source != 'source4':  # 근적외선 카메라 -> 검출 요소가 다르기 떄문에 제외(샴 모델로 해결 가능)
                preprocessed_video = video_preprocessor(path + '/' + v + '/' + source + '/video.avi')
                preprocessed_label = label_preprocessor(path + '/' + v + '/' + source + '/wave.csv',
                                                        preprocessed_video.shape[0] * preprocessed_video.shape[1])
                # n 블록 * 32 프레임
                return_dict[path[14:] + '_' + v + '_' + source] = {'preprocessed_video': preprocessed_video,
                                                                   'preprocessed_label': preprocessed_label}


if __name__ == '__main__':
    preprocessing(save_root_path="/test_data/",
                  dataset_root_path="/VIPL_HR/data",
                  train_ratio=0.8,
                  split_index=[0, 0],
                  index_train_name=0,
                  index_test_name=0
                  )

