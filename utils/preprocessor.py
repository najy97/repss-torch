# import package

from video_to_dataset import video_preprocessor
from csv_to_label import label_preprocessor
import multiprocessing
import h5py
import os


def preprocessing(save_root_path: str = "C:/preprocessed_dataset/",
                  dataset_root_path: str = "C:/VIPL_HR/data",
                  train_ratio: float = 0.8):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    data_list = [data for data in os.listdir(dataset_root_path)]  # p1 - p107

    process = []

    # multiprocessing
    for index, data_path in enumerate(data_list):
        proc = multiprocessing.Process(target=Preprocess_Dataset,
                                       args=(dataset_root_path + '/' + data_path, return_dict))
        process.append(proc)  # join할 프로세스 리스트 생성
        proc.start()  # 프로세스 시작 신호

    for proc in process:
        proc.join()  # 해당 프로세스가 종료될 때 까지 대기

    train = int(len(return_dict.keys()) * train_ratio)  # train/validation 분할

    index_name = 0
    for index, data_path in enumerate(return_dict.keys()[:train]):
        if index % 3 == 0:
            # hdf5파일 생성(열기) 및 쓰기
            train_file = h5py.File(save_root_path + 'train_' + str(index_name) + '.hdf5', "w")
            index_name += 1
        dset = train_file.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        if index != 0 and index % 2 == 0:
            train_file.close()
    if index % 2 != 0:
        train_file.close()

    index_name = 0
    for index, data_path in enumerate(return_dict.keys()[train:]):
        if index % 3 == 0:
            test_file = h5py.File(save_root_path + "test_" + str(index / 1700) + '.hdf5', "w")
            index_name += 1
        dset = test_file.create_group(data_path)
        dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
        dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
        if index != 0 and index % 2 == 0:
            test_file.close()
    if index % 2 != 0:
        train_file.close()


def Preprocess_Dataset(path, return_dict):
    for v in os.listdir(path):
        for source in os.listdir(path + '/' + v):
            if source != 'source4':  # 근적외선 카메라 -> 검출 요소가 다르기 떄문에 제외(샴 모델로 해결 가능)
                preprocessed_video = video_preprocessor(path + '/' + v + '/' + source + '/video.avi')
                preprocessed_label = label_preprocessor(path + '/' + v + '/' + source + '/wave.csv',
                                                        preprocessed_video.shape[0] * preprocessed_video.shape[1])
                # n 블록 * 32 프레임
                return_dict[path[-3:] + '_' + v + '_' + source] = {'preprocessed_video': preprocessed_video,
                                                                   'preprocessed_label': preprocessed_label}


if __name__ == '__main__':
    preprocessing(save_root_path="C:/preprocessed_dataset/",
                  dataset_root_path="C:/VIPL_HR/data",
                  train_ratio=0.8)
