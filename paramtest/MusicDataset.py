from torch.utils.data import Dataset
import csv
import os
import torch
import statistics
import numpy as np
sample_num = 100

# 用于训练和测试数据都在一个文件夹内
class MusicDataOne(Dataset):
    def __init__(self, data_dir, label_file, transform=None, pic_len=256, start=0, total=3223):
        self.start = start
        self.total = total

        label_file = open(label_file, 'r')
        label_reader = csv.reader(label_file)
        reader_list = list(label_reader)
        create_labels = True
        for line in reader_list:
            if create_labels:
                self.labels = [list(map(int, list(map(float, line))))]
                create_labels = False
            else:
                self.labels.append(list(map(int, list(map(float, line)))))
        self.label_len = len(self.labels[0])
        label_file.close()

        self.file_names = os.listdir(path=data_dir)
        self.sample_len = pic_len * pic_len
        self.data_dir = data_dir
        self.pic_len = pic_len
        self.transform = transform

    def __len__(self):
        return (self.total - self.start) * sample_num

    def __getitem__(self, idx):
        idx = int((idx / sample_num + self.start) * sample_num)
        file = open(self.file_names[int(idx / sample_num)])
        rows = list(csv.reader(file))
        row = list(map(float, rows[idx % sample_num]))
        time_data = row[0: self.sample_len]
        freq_data = row[self.sample_len: self.sample_len * 2]
        freq_data[0] = row[self.sample_len * 2]
        time_data = torch.Tensor(time_data).view(1, self.pic_len, self.pic_len)
        freq_data = torch.Tensor(freq_data).view(1, self.pic_len, self.pic_len)
        data = torch.cat((time_data, freq_data))
        label = self.labels[int(idx / sample_num)]
        if self.transform:
            data = self.transform(data)
        file.close()
        return data, label


# 用于训练和测试数据在不同的文件夹内
class MusicDataTwo(Dataset):
    """docstring for MusicDataTwo"""

    def __init__(self, data_dir, label_file, transform=None, pic_len=256):
        super(MusicDataTwo, self).__init__()

        label_file = open(label_file, 'r')
        label_reader = csv.reader(label_file)
        reader_list = list(label_reader)
        create_labels = True
        for line in reader_list:
            if not line:
                continue
            if create_labels:
                self.labels = [list(map(int, list(map(float, line))))]
                create_labels = False
            else:
                self.labels.append(list(map(int, list(map(float, line)))))
        self.label_len = len(self.labels[0])
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
        file = open(self.file_names[int(idx / sample_num)])
        rows = list(csv.reader(file))
        row = list(map(float, rows[idx % sample_num]))
        time_data = row[0: self.sample_len]
        freq_data = row[self.sample_len: self.sample_len * 2]
        freq_data[0] = row[self.sample_len * 2]
        time_data = torch.Tensor(time_data).view(1, self.pic_len, self.pic_len)
        freq_data = torch.Tensor(freq_data).view(1, self.pic_len, self.pic_len)
        data = torch.cat((time_data, freq_data))
        label = self.labels[int(idx / sample_num)]
        if self.transform:
            data = self.transform(data)
        file.close()
        return data, label


def get_data(time_data, pic_len):
    sample_len = pic_len * pic_len
    freq_data = abs(np.fft.fft(time_data) / sample_len)
    freq_data[0] = statistics.mean(freq_data[1:])
    time_data = np.array(time_data).reshape(pic_len, pic_len)
    freq_data = np.array(freq_data).reshape(pic_len, pic_len)
    data = torch.Tensor([time_data, freq_data])
    return data


# 用于训练和测试数据在同一个文件内
class MusicDataThree(Dataset):
    def __init__(self, data_file, label_file, transform=None, pic_len=256, start=0, total=3223):
        super(MusicDataThree, self).__init__()

        label_file = open(label_file, 'r')
        label_reader = csv.reader(label_file)
        reader_list = list(label_reader)
        create_labels = True
        for line in reader_list:
            if not line:
                continue
            if create_labels:
                self.labels = [list(map(int, list(map(float, line))))]
                create_labels = False
            else:
                self.labels.append(list(map(int, list(map(float, line)))))
        label_file.close()

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
        row = self.rows[int(idx/sample_num)]
        interval = int((len(row) - self.sample_len) / (sample_num - 1))
        start = idx % sample_num * interval
        data = get_data(row[start: start+self.sample_len])
        if self.transform:
            data = self.transform(data)
        label = self.labels[int(idx/sample_num)]
        return data, label
