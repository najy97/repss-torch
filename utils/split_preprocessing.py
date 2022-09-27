from utils.preprocessor import preprocessing

flag = False
split_index = []
index_train_name = index_test_name = 0
for i in range(0, 10, 2):
    if flag:
        flag = False
        split_index.append(i)
        (index_train_name, index_test_name) = preprocessing(save_root_path="/preprocessed_dataset/",
                                                            dataset_root_path="/VIPL_HR/data",
                                                            train_ratio=0.8,
                                                            split_index=split_index,
                                                            index_train_name=index_train_name,
                                                            index_test_name=index_test_name)
        split_index = [i]
    else:
        flag = True
        split_index.append(i)
# print('p'+i+'~p107 preprocessing')
# preprocessing(save_root_path="/preprocessed_dataset/",
#               dataset_root_path="/VIPL_HR/data",
#               train_ratio=0.8,
#               split_index=[i, 107],
#               index_train_name=index_train_name,
#               index_test_name=index_test_name
#               )
