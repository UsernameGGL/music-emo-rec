import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import csv
import time
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MusicDataset import MusicDataThree

# time.sleep(3600*12)
train_num = 2578  # 用来训练的曲子数
pic_len = 256  # 图片长度
batch_size = 100
label_len = 18
epoch_num = 1
data_dir = 'E:/data/cal500/music-data-v4-back/'
label_file = 'E:/data/cal500/labels_v4_back.csv'
data_file = 'E:/data/cal500/music-data-v4.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

print('start')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)
        linear_len = int(((((pic_len - 4) / 2 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(6 * linear_len * linear_len, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 18)

        self.conv1_2 = nn.Conv2d(1, 6, 5)
        self.conv2_2 = nn.Conv2d(6, 6, 5)
        self.conv3_2 = nn.Conv2d(6, 6, 7)
        self.conv4_2 = nn.Conv2d(6, 6, 5)
        linear_len_2 = int(((((pic_len - 4) / 2 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len_2 = linear_len_2
        self.fc1_2 = nn.Linear(6 * linear_len_2 * linear_len_2, 500)
        self.fc2_2 = nn.Linear(500, 100)
        self.fc3_2 = nn.Linear(100, 20)
        self.fc4_2 = nn.Linear(20, 18)

        # self.conv1_3 = nn.Conv2d(1, 6, 5)
        # self.conv2_3 = nn.Conv2d(6, 6, 5)
        # self.conv3_3 = nn.Conv2d(6, 6, 7)
        # self.conv4_3 = nn.Conv2d(6, 6, 5)
        # linear_len_3 = int(((((pic_len - 4) / 2 - 4) / 2 - 6) / 2 - 4) / 2)
        # self.linear_len_3 = linear_len_3
        # self.fc1_3 = nn.Linear(6 * linear_len_3 * linear_len_3, 500)
        # self.fc2_3 = nn.Linear(500, 100)
        # self.fc3_3 = nn.Linear(100, 20)
        # self.fc4_3 = nn.Linear(20, 18)

    def forward(self, x):
        first = torch.unsqueeze(x[:, 0], 1)
        second = torch.unsqueeze(x[:, 1], 1)
        # third = x[:, 2]
        first = self.pool(F.relu(self.conv1(first)))
        first = self.pool(F.relu(self.conv2(first)))
        first = self.pool(F.relu(self.conv3(first)))
        first = self.pool(F.relu(self.conv4(first)))
        first = first.view(-1, 6 * self.linear_len * self.linear_len)
        first = F.relu(self.fc1(first))
        first = F.relu(self.fc2(first))
        first = F.relu(self.fc3(first))
        first = (self.fc4(first))

        second = self.pool(F.relu(self.conv1_2(second)))
        second = self.pool(F.relu(self.conv2_2(second)))
        second = self.pool(F.relu(self.conv3_2(second)))
        second = self.pool(F.relu(self.conv4_2(second)))
        second = second.view(-1, 6 * self.linear_len * self.linear_len)
        second = F.relu(self.fc1_2(second))
        second = F.relu(self.fc2_2(second))
        second = F.relu(self.fc3_2(second))
        second = (self.fc4_2(second))

        # third = self.pool(F.relu(self.conv1_3(third)))
        # third = self.pool(F.relu(self.conv2_3(third)))
        # third = self.pool(F.relu(self.conv3_3(third)))
        # third = self.pool(F.relu(self.conv4_3(third)))
        # third = third.view(-1, 6 * self.linear_len * self.linear_len)
        # third = F.relu(self.fc1_3(third))
        # third = F.relu(self.fc2_3(third))
        # third = F.relu(self.fc3_3(third))
        # third = F.relu(self.fc4_3(third))
        # return first + second + third
        y = first + second
        return y


net = Net()
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
net.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min')


def train():
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        train_loader = DataLoader(dataset=MusicDataThree(data_file, label_file, total=train_num),
                                  batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # outputs = torch.round(outputs)
            loss = criterion(outputs, labels)
            ######################################################
            scheduler.step(loss)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), '2 0+coon-meanfreq.pt')


def test():
    correct = 0
    correct_v2 = 0
    total = 0
    loss = 0
    sigmoid = nn.Sigmoid()
    cnt = 0
    my_total = 0
    correct_v3 = 0
    correct_v4 = 0
    total_v4 = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=MusicDataThree(data_file, label_file, start=train_num),
                                 batch_size=batch_size, shuffle=True)
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = sigmoid(outputs)
            cnt += 1

            for k in range(batch_size):
                _, index = torch.sort(outputs[k], descending=True)
                emotion_num = int(torch.sum(labels[k]).item())
                total_v4 += emotion_num
                for kk in range(emotion_num):
                    if labels[k][index[kk]] == 1:
                        correct_v4 += 1
            outputs = torch.round(outputs)
            total += labels.size(0)
            #########################################################
            my_total += labels[labels == 1].sum().item()
            correct += (outputs.data == labels).sum().item()
            tmp_loss = abs(outputs.data - labels).sum().item()
            if tmp_loss == 0:
                correct_v2 += 1
            loss += tmp_loss
            for k in range(batch_size):
                for kk in range(label_len):
                    if labels[k][kk] == 1 and outputs[k][kk] == 1:
                        correct_v3 += 1
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total / 18))
    print('Loss of the network: {}'.format(loss))
    print('My_Accuracy of the network on the test images: %d %%' % (
            100 * correct_v2 / total))
    print('My_Accuracy_2 of the network on the test images: %d %%' % (
            100 * correct_v3 / my_total))
    print('My_Accuracy_4 of the network on the test images: %d %%' % (
            100 * correct_v4 / total_v4))


if __name__ == "__main__":
    train()
    test()
