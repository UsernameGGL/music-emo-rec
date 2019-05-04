from torch.utils.data import Dataset
import csv
import os
import torch
import statistics
import numpy as np

sample_num = 100
total = 3219
start = 0
basic_dir = 'E:/data/cal500/'
data_dir = basic_dir + 'music-data-v4-back/'
label_file = basic_dir + 'labels_v4_back.csv'
data_file = basic_dir + 'music-data-v4.csv'


def one_hot_label(label_file):
    label_file = open(label_file, 'r')
    label_reader = csv.reader(label_file)
    reader_list = list(label_reader)
    label_file.close()
    return torch.Tensor(reader_list[0])


def get_label(label_file):
    label_file = open(label_file, 'r')
    label_reader = csv.reader(label_file)
    reader_list = list(label_reader)
    create_labels = True
    for line in reader_list:
        if not line:
            continue
        if create_labels:
            labels = [list(map(int, list(map(float, line))))]
            create_labels = False
        else:
            labels.append(list(map(int, list(map(float, line)))))
    return torch.Tensor(labels)


def get_data(time_data, pic_len, transform):
    sample_len = pic_len * pic_len
    time_data = list(map(float, time_data))
    freq_data = abs(np.fft.fft(time_data) / sample_len)
    freq_data[0] = statistics.mean(freq_data[1:])
    time_data = np.array(time_data).reshape(pic_len, pic_len)
    freq_data = np.array(freq_data).reshape(pic_len, pic_len)
    data = torch.Tensor([time_data, freq_data])
    if transform:
        data = transform(data)
    return data


def from_file(data_file, sample_idx, pic_len, transform):
    sample_len = pic_len * pic_len

    file = open(data_file)
    rows = list(csv.reader(file))
    row = rows[sample_idx]
    time_data = list(map(float, row[0: sample_len]))
    freq_data = row[sample_len: sample_len * 2]
    freq_data[0] = row[sample_len * 2]
    time_data = torch.Tensor(time_data).view(1, pic_len, pic_len)
    freq_data = torch.Tensor(freq_data).view(1, pic_len, pic_len)
    data = torch.cat((time_data, freq_data))
    file.close()
    if transform:
        data = transform(data)
    return data


# 用于训练和测试数据都在一个文件夹内, 注意total的作用其实是end，也仅是数据结尾下标
class MusicDataOne(Dataset):
    def __init__(self, data_dir=data_dir, label_file=label_file, transform=None, pic_len=256, start=start, total=total):
        self.start = start
        self.total = total

        self.labels = get_label(label_file)

        self.file_names = os.listdir(path=data_dir)
        self.sample_len = pic_len * pic_len
        self.data_dir = data_dir
        self.pic_len = pic_len
        self.transform = transform

    def __len__(self):
        return (self.total - self.start) * sample_num

    def __getitem__(self, idx):
        idx = int((idx / sample_num + self.start) * sample_num)
        music_idx = int(idx / sample_num)
        sample_idx = idx % sample_num
        data = from_file(data_file, sample_idx, self.pic_len, self.transform)
        label = self.labels[music_idx]
        return data, label


# 用于训练和测试数据在不同的文件夹内
class MusicDataTwo(Dataset):
    """docstring for MusicDataTwo"""

    def __init__(self, data_dir=data_dir, label_file=label_file, transform=None, pic_len=256):
        super(MusicDataTwo, self).__init__()

        self.labels = get_label(label_file)
        self.len = len(self.labels)
        label_file.close()

        self.file_names = os.listdir(path=data_dir)
        self.sample_len = pic_len * pic_len
        self.data_dir = data_dir
        self.pic_len = pic_len
        self.transform = transform

    def __len__(self):
        return self.len * sample_num

    def __getitem__(self, idx):
        music_idx = int(idx / sample_num)
        sample_idx = idx % sample_num
        data = from_file(self.file_names[music_idx], sample_idx, self.pic_len, self.transform)
        label = self.labels[music_idx]
        return data, label


# 用于训练和测试数据在同一个文件内
class MusicDataThree(Dataset):
    def __init__(self, data_file=data_file, label_file=label_file,
     transform=None, pic_len=256, start=start, total=total, mode='normal'):
        super(MusicDataThree, self).__init__()

        if mode == 'one-hot':
            self.labels = one_hot_label(label_file)
        else:
            self.labels = get_label(label_file)

        data_file = open(data_file, 'r')
        reader = csv.reader(data_file)
        self.rows = list(reader)
        self.len = total - start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform
        self.start = start

    def __len__(self):
        return self.len * sample_num

    def __getitem__(self, idx):
        idx = int((idx / sample_num + self.start) * sample_num)
        music_idx = int(idx / sample_num)
        sample_idx = idx % sample_num
        row = self.rows[music_idx]
        interval = int((len(row) - self.sample_len) / (sample_num - 1))
        start = sample_idx * interval
        data = get_data(row[start: start + self.sample_len], self.pic_len, self.transform)
        label = self.labels[music_idx]
        return data, label


class ExpNorm(object):
    """docstring for ExpNorm"""
    def __init__(self):
        super(ExpNorm, self).__init__()


    def __call__(self, data):
        time_data = np.array(data[0])
        freq_data = np.array(data[1])
        time_data = (time_data - np.mean(time_data)) / np.std(time_data)
        freq_data = (freq_data - np.mean(freq_data)) / np.std(freq_data)
        return torch.Tensor([time_data, freq_data])


class IniNorm(object):
    """docstring for IniNorm"""
    def __init__(self, arg):
        super(IniNorm, self).__init__()
    

    def __call__(set, data):
        time_data = np.array(data[0])
        freq_data = np.array(data[1])
        time_data = (time_data - np.min(time_data)) / (np.max(time_data) - np.min(time_data))
        freq_data = (freq_data - np.min(freq_data)) / (np.max(freq_data) - np.min(freq_data))
        
