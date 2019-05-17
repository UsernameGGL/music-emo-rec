from torch.utils.data import Dataset
import csv
import os
import torch
# import statistics
import numpy as np
import random

sample_num = 100
total = 3219
start = 0
basic_dir = '../'
data_dir = basic_dir + 'raw-data-v5/'
label_file = basic_dir + 'labels-v5.csv'
data_file = basic_dir + 'music-data-v8.csv'


def one_hot_label(label_file):
    label_file = open(label_file, 'r')
    label_reader = csv.reader(label_file)
    reader_list = list(label_reader)
    label_file.close()
    return torch.LongTensor(list(map(int, reader_list[0])))


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
    freq_data = abs(np.fft.fft(time_data) / sample_len)
    freq_data[0] = freq_data[1]
    time_data = np.array(time_data).reshape(pic_len, pic_len)
    freq_data = np.array(freq_data).reshape(pic_len, pic_len)*200
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
        self.rows = data_file.readlines()
        self.len = total - start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform
        self.start = start

    def __len__(self):
        # return self.len * sample_num
        return self.len

    def __getitem__(self, idx):
        row = self.rows[idx].split(',')
        # length = len(row)
        # row[length - 1] = row[length - 1].split('\n')[0]
        # start = random.randint(0, length - self.sample_len)
        start = 0
        row = list(map(float, row[start: start + self.sample_len]))
        data = get_data(row, self.pic_len, self.transform)
        label = self.labels[idx]
        return data, label


# 用于训练和测试数据在同一个文件夹内，但每一行未处理的raw_data都属于单独一个文件
class MusicDataFour(Dataset):
    def __init__(self, data_dir=data_dir, label_file=label_file,
     transform=None, pic_len=256, start=start, total=total, mode='normal'):
        super(MusicDataFour, self).__init__()

        if mode == 'one-hot':
            self.labels = one_hot_label(label_file)
        else:
            self.labels = get_label(label_file)


        self.file_names = os.listdir(path=data_dir)
        self.len = total - start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform
        self.start = start
        self.data_dir = data_dir

    def __len__(self):
        # return self.len * sample_num
        return self.len

    def __getitem__(self, idx):
        # idx = int((idx / sample_num + self.start) * sample_num)
        # music_idx = int(idx / sample_num)
        # sample_idx = idx % sample_num
        # row = list(csv.reader(open(self.data_dir+self.file_names[music_idx], 'r')))[0]
        row = list(csv.reader(open(self.data_dir+self.file_names[idx], 'r')))[0]
        # length = len(row)
        # interval = int((length - self.sample_len) / (sample_num - 1))
        # start = sample_idx * interval
        # start = random.randint(0, length - self.sample_len)
        # row = list(map(int, row[start: start + self.sample_len]))
        row = list(map(int, row[0: self.sample_len]))
        data = get_data(row, self.pic_len, self.transform)
        # label = self.labels[music_idx]
        label = self.labels[idx]
        return data, label


class Musicdata_v7(Dataset):
    """docstring for Musicdata_v7"""
    def __init__(self, data_file, label_file=label_file,
     transform=None, start=start, total=total, mode='normal'):
        super(Musicdata_v7, self).__init__()
        data_file = open(data_file, 'r')
        self.rows = list(csv.reader(data_file))
        self.len = total - start
        self.transform = transform
        self.start = start
        self.labels = get_label(label_file)


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        data = torch.Tensor(list(map(float, self.rows[idx]))[0: 16]).view(1, 4, 4)
        label = self.labels[idx]
        return data, label


class Musicdata_v9(Dataset):
    """docstring for Musicdata_v9"""
    def __init__(self, data_file=basic_dir + 'music-data-v9.csv', label_file=label_file,
     transform=None, start=start, total=total, mode='normal'):
        super(Musicdata_v9, self).__init__()
        data_file = open(data_file, 'r')
        self.rows = list(csv.reader(data_file))
        self.len = total - start
        self.transform = transform
        self.start = start
        self.labels = get_label(label_file)


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        data = torch.Tensor(list(map(float, self.rows[idx]))[0: 64]).view(1, 8, 8)
        label = self.labels[idx]
        return data, label


class Musicdata_LSTM(Dataset):
    """docstring for Musicdata_LSTM"""
    def __init__(self, data_file, label_file=label_file,
     transform=None, start=start, total=total, mode='deep'):
        super(Musicdata_LSTM, self).__init__()
        data_file = open(data_file, 'r')
        self.rows = list(csv.reader(data_file))
        self.len = total - start
        self.transform = transform
        self.start = start
        self.labels = get_label(label_file)
        if mode == 'deep':
            self.data_len = 65536
        else:
            self.data_len = 16


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        data = torch.Tensor(list(map(float, self.rows[idx]))[0: self.data_len]).view(1, self.data_len)
        label = self.labels[idx].view(1, 18)
        return data, label


class Musicdata_Bin(Dataset):
    def __init__(self, data_file, label_file=label_file,
     transform=None, start=start, total=total, clsf_idx=0):
        data_file = open(data_file, 'r')
        self.rows = list(csv.reader(data_file))
        self.len = total - start
        self.transform = transform
        self.start = start
        self.labels = get_label(label_file)
        self.clsf_idx = clsf_idx


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        length = len(self.rows[idx])
        start = random.randint(0, length - 16)
        data = torch.Tensor(list(map(float, self.rows[idx]))[start: start + 16]).view(1, 4, 4)
        label = (self.labels[idx][self.clsf_idx]).long()
        # label = torch.LongTensor([label, 1-label])
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
    def __init__(self):
        super(IniNorm, self).__init__()
    

    def __call__(set, data):
        time_data = np.array(data[0])
        freq_data = np.array(data[1])
        time_data = (time_data - np.min(time_data)) / (np.max(time_data) - np.min(time_data))
        freq_data = (freq_data - np.min(freq_data)) / (np.max(freq_data) - np.min(freq_data))
        return torch.Tensor([time_data, freq_data])
        
