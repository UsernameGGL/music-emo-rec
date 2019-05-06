import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import sys
from torch.utils.data import DataLoader
from MusicDataset import MusicDataThree
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_start = 0
train_end = 128  # 用来训练的曲子数
test_start = 0
test_end = 128
batch_size = 128
pic_len = 256
label_len = 18
epoch_num = 10000
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
basic_dir = 'E:/data/cal500/'
data_dir = basic_dir + 'music-data-v4-back/'
label_file = basic_dir + 'labels_v4_back.csv'
data_file = basic_dir + 'music-data-v4.csv'
basic_dir = '../'
# basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
record_file = 'record-one-hot.txt'


class Justreducelr_0(nn.Module):
    def __init__(self):
        super(Justreducelr_0, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.norm1 = nn.GroupNorm(3, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.GroupNorm(4, 16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.norm3 = nn.GroupNorm(4, 8)
        linear_len = int(((pic_len - 4) / 2 - 4) / 2 - 6 - 4)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(8 * linear_len * linear_len, 500)
        self.norm4 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 100)
        self.norm5 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 18)
        self.fc5 = nn.Linear(18, 18)

    def forward(self, x):
        x = self.norm1(self.pool(F.relu(self.conv1(x))))
        x = self.norm2(self.pool(F.relu(self.conv2(x))))
        x = self.norm2(F.relu(self.conv3(x)))
        x = self.norm3(F.relu(self.conv4(x)))
        x = x.view(-1, 8 * self.linear_len * self.linear_len)
        x = self.norm4(self.fc1(x))
        x = self.norm5(self.fc2(x))
        x = (self.fc3(x))
        x = self.fc4(x)
        x = (self.fc5(x))
        return x


class Justreducelr_0_ini(nn.Module):
    def __init__(self):
        super(Justreducelr_0_ini, self).__init__()
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


net = Justreducelr_0()

# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()  # ##################################################### here
train_set = MusicDataThree(data_file, label_file, start=train_start, total=train_end)
test_set = MusicDataThree(data_file, label_file, start=test_start, total=test_end)
criterion = nn.BCEWithLogitsLoss()


def set_record_file(file_name):
    global record_file
    record_file = file_name


def trans_mode(mode='normal'):
    global criterion
    global train_set
    global test_set
    if mode == 'one-hot':
        label_file = basic_dir + 'one-hot-label.csv'
        train_set = MusicDataThree(data_file, label_file,start=train_start, total=train_end)
        test_set = MusicDataThree(data_file, label_file, start=test_start, total=test_end)
        criterion = nn.CrossEntropyLoss()


def train(net=net, criterion=criterion, model_path='tmp.pt', dataset=train_set, optimizer=None, scheduler=None):
    with open(record_file, 'a') as f:
        f.write('This is model of' + model_path + '\n')

    net.to(device)
    if not optimizer:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.1)
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, 
            threshold=1e-6, factor=0.5, min_lr=1e-7)
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        train_loader = DataLoader(dataset=dataset,
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
            if epoch % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, epoch + 1, running_loss / 10))
                with open(record_file, 'a') as f:
                    f.write('[%d, %5d] loss: %.3f\n' %
                    (epoch + 1, epoch + 1, running_loss / 10))
                running_loss = 0.0
                torch.save(net.state_dict(), model_path)

    print('Finished Training')
    torch.save(net.state_dict(), model_path)
    return net


def test(net, net_name, dataset=test_set):
    correct = 0
    correct_v2 = 0
    total = 0
    loss = 0
    sigmoid = nn.Sigmoid()
    my_total = 0
    correct_v3 = 0
    correct_v4 = 0
    total_v4 = 0
    threshold = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, shuffle=True)
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            # print(1, outputs)
            # ####################################################here
            outputs = sigmoid(outputs)
            # print(2, outputs)
            for k in range(batch_size):
                _, index = torch.sort(outputs[k], descending=True)
                emotion_num = int(torch.sum(labels[k]).item())
                total_v4 += emotion_num
                for kk in range(emotion_num):
                    if labels[k][index[kk]] == 1:
                        correct_v4 += 1
            print(torch.min(outputs))
            print(torch.max(outputs))
            one_correct = 0
            for i in np.arange(0.3, 0.7, 0.01):
                tmp_outputs = outputs.clone()
                tmp_outputs[tmp_outputs > i] = 1
                tmp_outputs[tmp_outputs <= i] = 0
                tmp_correct = (tmp_outputs.data == labels).sum().item()
                if tmp_correct > one_correct:
                    one_correct = tmp_correct
                    threshold = i
                    r_outputs = tmp_outputs.clone()
            print('--------threshold', threshold)
            outputs = r_outputs
            # outputs = torch.round(outputs)
            total += labels.size(0)
            ########################################
            my_total += labels[labels == 1].sum().item()
            correct += one_correct
            # correct += (outputs.data == labels).sum().item()
            # loss += abs(outputs.data - labels).sum().item()
            tmp_loss = abs(outputs.data - labels).sum().item()
            if tmp_loss == 0:
                correct_v2 += 1
            loss += tmp_loss
            ####################################
            for k in range(batch_size):
                for kk in range(label_len):
                    if labels[k][kk] == 1 and outputs[k][kk] == 1:
                        correct_v3 += 1
            print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total / 18))
            print('Loss of the network: {}'.format(loss))
            print('My_Accuracy of the network on the test images: %f %%' % (
                    100 * correct_v2 / total))
            print('My_Accuracy_2 of the network on the test images: %f %%' % (
                    100 * correct_v3 / my_total))
            print('My_Accuracy_4 of the network on the test images: %f %%' % (
                    100 * correct_v4 / total_v4))

    print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total / 18))
    print('Loss of the network: {}'.format(loss))
    print('My_Accuracy of the network on the test images: %f %%' % (
            100 * correct_v2 / total))
    print('My_Accuracy_2 of the network on the test images: %f %%' % (
            100 * correct_v3 / my_total))
    print('My_Accuracy_4 of the network on the test images: %f %%' % (
            100 * correct_v4 / total_v4))
    with open(record_file, 'a') as f:
        f.write('This is the result of' + net_name + '\n')
        f.write('Accuracy of the network on the test images: %f %%\n' % (
            100 * correct / total / 18))
        f.write('Loss of the network: {}\n'.format(loss))
        f.write('My_Accuracy of the network on the test images: %f %%\n' % (
            100 * correct_v2 / total))
        f.write('My_Accuracy_4 of the network on the test images: %f %%\n' % (
            100 * correct_v4 / total_v4))


class Norm_0_1(nn.Module):
    def __init__(self):
        super(Norm_0_1, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.norm3 = nn.BatchNorm2d(8)
        linear_len = int(((pic_len - 4) / 2 - 4) / 2 - 6 - 4)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(8 * linear_len * linear_len, 500)
        self.norm4 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm5 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm6 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm7 = nn.BatchNorm1d(num_features=18)
        self.fc5 = nn.Linear(18, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.norm2(x)
        x = (F.relu(self.conv3(x)))
        x = self.norm2(x)
        x = (F.relu(self.conv4(x)))
        x = self.norm3(x)
        x = x.view(-1, 8 * self.linear_len * self.linear_len)
        x = F.relu(self.fc1(x))
        x = self.norm4(x)
        x = F.relu(self.fc2(x))
        x = self.norm5(x)
        x = F.relu(self.fc3(x))
        x = self.norm6(x)
        x = self.fc4(x)
        x = self.norm7(x)
        x = (self.fc5(x))
        return x


class Coon_0_2(nn.Module):
    def __init__(self):
        super(Coon_0_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)
        linear_len = int(((((pic_len - 4) / 2 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len = linear_len
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.conv6 = nn.Conv2d(6, 12, 3)
        self.conv7 = nn.Conv2d(12, 18, 3)

    def forward(self, x):
        first = torch.unsqueeze(x[:, 0], 1)
        second = torch.unsqueeze(x[:, 1], 1)
        # third = x[:, 2]
        first = self.pool(self.conv1(first))
        first = self.pool(self.conv2(first))
        first = self.pool(self.conv3(first))
        first = self.pool(self.conv4(first))
        for i in range(3):
            first = F.relu(self.conv5(first))
        first = self.conv6(first)
        first = self.conv7(first)

        second = self.pool(self.conv1(second))
        second = self.pool(self.conv2(second))
        second = self.pool(self.conv3(second))
        second = self.pool(self.conv4(second))
        for i in range(3):
            second = F.relu(self.conv5(second))
        second = self.conv6(second)
        second = self.conv7(second)
        y = (first + second).view(-1, label_len)
        return y


class Deeper_1_3(nn.Module):
    def __init__(self):
        super(Deeper_1_3, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.norm3 = nn.BatchNorm2d(8)
        linear_len = int(((pic_len - 4) / 2 - 2 * 10 - 4) / 2 - 6 - 4)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(8 * linear_len * linear_len, 500)
        self.norm4 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm5 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm6 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm7 = nn.BatchNorm1d(num_features=18)
        self.fc5 = nn.Linear(18, 18)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.norm1(x)
        for i in range(10):
            x = F.relu(self.conv5(x))
            x = self.norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.norm2(x)
        x = (F.relu(self.conv3(x)))
        x = self.norm2(x)
        x = (F.relu(self.conv4(x)))
        x = self.norm3(x)
        x = x.view(-1, 8 * self.linear_len * self.linear_len)
        x = F.relu(self.fc1(x))
        x = self.norm4(x)
        x = F.relu(self.fc2(x))
        x = self.norm5(x)
        x = F.relu(self.fc3(x))
        x = self.norm6(x)
        x = self.fc4(x)
        x = self.norm7(x)
        x = (self.fc5(x))
        return x


class Coon_1_4(nn.Module):
    def __init__(self):
        super(Coon_1_4, self).__init__()
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
        self.norm3 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm4 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm5 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm6 = nn.BatchNorm1d(num_features=18)

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
        first = self.norm1(self.pool(F.relu(self.conv1(first))))
        first = self.norm1(self.pool(F.relu(self.conv2(first))))
        first = self.norm1(self.pool(F.relu(self.conv3(first))))
        first = self.norm1(self.pool(F.relu(self.conv4(first))))
        first = first.view(-1, 6 * self.linear_len * self.linear_len)
        first = self.norm3(F.relu(self.fc1(first)))
        first = self.norm4(F.relu(self.fc2(first)))
        first = self.norm5(F.relu(self.fc3(first)))
        first = (self.fc4(first))

        second = self.norm1(self.pool(F.relu(self.conv1_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv2_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv3_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv4_2(second))))
        second = second.view(-1, 6 * self.linear_len * self.linear_len)
        second = self.norm3(F.relu(self.fc1_2(second)))
        second = self.norm4(F.relu(self.fc2_2(second)))
        second = self.norm5(F.relu(self.fc3_2(second)))
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


class Deeper_4_5(nn.Module):
    def __init__(self):
        super(Deeper_4_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)
        linear_len = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(6 * linear_len * linear_len, 500)
        self.norm3 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm4 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm5 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm6 = nn.BatchNorm1d(num_features=18)

        self.conv1_2 = nn.Conv2d(1, 6, 3)
        self.conv5_2 = nn.Conv2d(6, 6, 3)
        self.conv2_2 = nn.Conv2d(6, 6, 5)
        self.conv3_2 = nn.Conv2d(6, 6, 7)
        self.conv4_2 = nn.Conv2d(6, 6, 5)
        linear_len_2 = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
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
        first = self.norm1(self.pool(F.relu(self.conv1(first))))
        for i in range(5):
            first = self.norm1(F.relu(self.conv5(first)))
        first = self.norm1(self.pool(F.relu(self.conv2(first))))
        first = self.norm1(self.pool(F.relu(self.conv3(first))))
        first = self.norm1(self.pool(F.relu(self.conv4(first))))
        first = first.view(-1, 6 * self.linear_len * self.linear_len)
        first = self.norm3(F.relu(self.fc1(first)))
        first = self.norm4(F.relu(self.fc2(first)))
        first = self.norm5(F.relu(self.fc3(first)))
        first = (self.fc4(first))

        second = self.norm1(self.pool(F.relu(self.conv1_2(second))))
        for i in range(5):
            second = self.norm1(F.relu(self.conv5_2(second)))
        second = self.norm1(self.pool(F.relu(self.conv2_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv3_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv4_2(second))))
        second = second.view(-1, 6 * self.linear_len * self.linear_len)
        second = self.norm3(F.relu(self.fc1_2(second)))
        second = self.norm4(F.relu(self.fc2_2(second)))
        second = self.norm5(F.relu(self.fc3_2(second)))
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


class Expnorm_5_6(nn.Module):
    def __init__(self):
        super(Expnorm_5_6, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)
        linear_len = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(6 * linear_len * linear_len, 500)
        self.norm3 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm4 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm5 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm6 = nn.BatchNorm1d(num_features=18)

        self.conv1_2 = nn.Conv2d(1, 6, 3)
        self.conv5_2 = nn.Conv2d(6, 6, 3)
        self.conv2_2 = nn.Conv2d(6, 6, 5)
        self.conv3_2 = nn.Conv2d(6, 6, 7)
        self.conv4_2 = nn.Conv2d(6, 6, 5)
        linear_len_2 = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
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
        first = self.norm1(self.pool(F.relu(self.conv1(first))))
        for i in range(5):
            first = self.norm1(F.relu(self.conv5(first)))
        first = self.norm1(self.pool(F.relu(self.conv2(first))))
        first = self.norm1(self.pool(F.relu(self.conv3(first))))
        first = self.norm1(self.pool(F.relu(self.conv4(first))))
        first = first.view(-1, 6 * self.linear_len * self.linear_len)
        first = self.norm3(F.relu(self.fc1(first)))
        first = self.norm4(F.relu(self.fc2(first)))
        first = self.norm5(F.relu(self.fc3(first)))
        first = (self.fc4(first))

        second = self.norm1(self.pool(F.relu(self.conv1_2(second))))
        for i in range(5):
            second = self.norm1(F.relu(self.conv5_2(second)))
        second = self.norm1(self.pool(F.relu(self.conv2_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv3_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv4_2(second))))
        second = second.view(-1, 6 * self.linear_len * self.linear_len)
        second = self.norm3(F.relu(self.fc1_2(second)))
        second = self.norm4(F.relu(self.fc2_2(second)))
        second = self.norm5(F.relu(self.fc3_2(second)))
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


class Ininorm_5_6(nn.Module):
    def __init__(self):
        super(Ininorm_5_6, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(6, 6, 7)
        self.conv4 = nn.Conv2d(6, 6, 5)
        linear_len = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
        self.linear_len = linear_len
        self.fc1 = nn.Linear(6 * linear_len * linear_len, 500)
        self.norm3 = nn.BatchNorm1d(num_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.norm4 = nn.BatchNorm1d(num_features=100)
        self.fc3 = nn.Linear(100, 20)
        self.norm5 = nn.BatchNorm1d(num_features=20)
        self.fc4 = nn.Linear(20, 18)
        self.norm6 = nn.BatchNorm1d(num_features=18)

        self.conv1_2 = nn.Conv2d(1, 6, 3)
        self.conv5_2 = nn.Conv2d(6, 6, 3)
        self.conv2_2 = nn.Conv2d(6, 6, 5)
        self.conv3_2 = nn.Conv2d(6, 6, 7)
        self.conv4_2 = nn.Conv2d(6, 6, 5)
        linear_len_2 = int(((((pic_len - 2) / 2 - 2 * 5 - 4) / 2 - 6) / 2 - 4) / 2)
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
        first = self.norm1(self.pool(F.relu(self.conv1(first))))
        for i in range(5):
            first = self.norm1(F.relu(self.conv5(first)))
        first = self.norm1(self.pool(F.relu(self.conv2(first))))
        first = self.norm1(self.pool(F.relu(self.conv3(first))))
        first = self.norm1(self.pool(F.relu(self.conv4(first))))
        first = first.view(-1, 6 * self.linear_len * self.linear_len)
        first = self.norm3(F.relu(self.fc1(first)))
        first = self.norm4(F.relu(self.fc2(first)))
        first = self.norm5(F.relu(self.fc3(first)))
        first = (self.fc4(first))

        second = self.norm1(self.pool(F.relu(self.conv1_2(second))))
        for i in range(5):
            second = self.norm1(F.relu(self.conv5_2(second)))
        second = self.norm1(self.pool(F.relu(self.conv2_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv3_2(second))))
        second = self.norm1(self.pool(F.relu(self.conv4_2(second))))
        second = second.view(-1, 6 * self.linear_len * self.linear_len)
        second = self.norm3(F.relu(self.fc1_2(second)))
        second = self.norm4(F.relu(self.fc2_2(second)))
        second = self.norm5(F.relu(self.fc3_2(second)))
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
