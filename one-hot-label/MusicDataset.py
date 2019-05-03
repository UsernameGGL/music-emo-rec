from torch.utils.data import Dataset
import csv
import os
import torch
import statistics
import numpy as np
sample_num = 100


def get_label(label_file):
    label_file = open(label_file, 'r')
    label_reader = csv.reader(label_file)
    reader_list = list(label_reader)
    label_file.close()
    return reader_list[0]

def get_data(time_data, pic_len):
    sample_len = pic_len * pic_len
    freq_data = abs(np.fft.fft(time_data) / sample_len)
    freq_data[0] = statistics.mean(freq_data[1:])
    time_data = np.array(time_data).reshape(pic_len, pic_len)
    freq_data = np.array(freq_data).reshape(pic_len, pic_len)
    data = torch.Tensor([time_data, freq_data])
    return data


def from_file(data_file, sample_idx, pic_len):
    sample_len = pic_len * pic_len

    file = open(data_file)
    rows = list(csv.reader(file))
    row = list(map(float, rows[sample_idx]))
    time_data = row[0: sample_len]
    freq_data = row[sample_len: sample_len * 2]
    freq_data[0] = row[sample_len * 2]
    time_data = torch.Tensor(time_data).view(1, pic_len, pic_len)
    freq_data = torch.Tensor(freq_data).view(1, pic_len, pic_len)
    data = torch.cat((time_data, freq_data))
    file.close()
    return data


# 用于训练和测试数据在同一个文件内
class MusicDataOne(Dataset):
    def __init__(self, data_file, label_file, transform=None, pic_len=256, start=0, total=3223):
        super(MusicDataOne, self).__init__()

        self.labels = get_label(label_file)

        data_file = open(data_file, 'r')
        reader = csv.reader(data_file)
        self.rows = list(reader)
        self.len = total - start
        self.start = start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform

    def __len__(self):
        return self.len * sample_num

    def __getitem__(self, idx):
        idx = int((idx / sample_num + self.start) * sample_num)
        music_idx = int(idx / sample_num)
        sample_idx = idx % sample_num
        row = self.rows[music_idx]
        interval = int((len(row) - self.sample_len) / (sample_num - 1))
        start = sample_idx * interval
        data = get_data(row[start: start+self.sample_len], self.pic_len)
        if self.transform:
            data = self.transform(data)
        label = self.labels[music_idx]
        return data, label


# 用于训练和测试数据在同一个文件夹内
class MusicDataTwo(Dataset):
    def __init__(self, data_dir, label_file, transform=None, pic_len=256, start=0, total=3223):
        super(MusicDataTwo, self).__init__()

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
        data = from_file(self.file_names[music_idx], sample_idx, self.pic_len)
        label = self.labels[music_idx]
        if self.transform:
            data = self.transform(data)
        return data, label
        
