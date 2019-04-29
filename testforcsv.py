import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import csv
import time
import _thread
import statistics
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
# 重写Dataset类
class TimeFreqDataset(Dataset):

    def __init__(self, data, label):
        self.len = len(data)
        self.data = transform(data)
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len
pic_len = 256
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        linear_len = int(((pic_len - 4) / 2 - 4) / 2 - 6 - 4)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(8 * linear_len * linear_len, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 18)
        self.fc5 = nn.Linear(18, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = x.view(-1, 8 * self.linear_len * self.linear_len)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = (self.fc5(x))
        return x
net = Net()

