from os import listdir
from os.path import isfile, join
import glob
from collections import Counter
import pandas as pd


def prepare_colon_tma_manual_data():
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = '/home/compu/jh/data/colon_tma/COLON_MANUAL_512'

    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir, parse_label=True,label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir, parse_label=True,label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir, parse_label=True,label_value=0)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017
    return train_set, valid_set, test_set


train_set, valid_set, test_set = prepare_colon_tma_manual_data()
train_set = pd.DataFrame(train_set, columns=['path', 'class'])
valid_set = pd.DataFrame(valid_set, columns=['path', 'class'])
test_set = pd.DataFrame(test_set, columns=['path', 'class'])
train_set.to_csv('/home/compu/jh/data/colon_tma/COLON_MANUAL_512/train.csv', index=False)
valid_set.to_csv('/home/compu/jh/data/colon_tma/COLON_MANUAL_512/valid.csv', index=False)
test_set.to_csv('/home/compu/jh/data/colon_tma/COLON_MANUAL_512/test.csv', index=False)