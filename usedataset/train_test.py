import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import sys
from torch.utils.data import DataLoader
from MusicDataset import MusicDataThree
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_start = 0
train_end = 2560  # 用来训练的曲子数
test_start = 2560
test_end = 3219
batch_size = 128
pic_len = 256
label_len = 18
epoch_num = 125
record_cnt = 10
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
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 12, 3)
        self.conv4 = nn.Conv2d(12, 15, 3)
        self.conv5 = nn.Conv2d(15, 18, 3)
        self.norm1 = nn.GroupNorm(3, 6)
        self.norm2 = nn.GroupNorm(4, 12)
        self.norm3 = nn.GroupNorm(5, 15)
        self.norm4 = nn.GroupNorm(6, 18)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU(0.1)
        # linear_len = int(((pic_len - 4) / 2 - 4) / 2 - 6 - 4)
        # self.linear_len = linear_len

    def forward(self, x):
        # x = self.norm1(self.pool(F.relu(self.conv1(x))))
        # for i in range(2):
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        # x = self.pool(F.relu(self.norm1(self.conv2(x))))
        for i in range(60):
            x = self.relu(self.norm1(self.conv2(x)))
        x = self.relu(self.norm2(self.conv3(x)))
        x = self.relu(self.norm3(self.conv4(x)))
        x = self.relu(self.norm4(self.conv5(x)))
        return x.view(-1, 18)


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


class SimpleCNN(nn.Module):
    """docstring for SimpleCNN"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(6, 12, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(12, 18, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(18, 24, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(24, 30, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(30, 24, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(24, 18, 2)
            )
        self.fc = nn.Sequential(
                nn.Linear(18, 36),
                nn.ReLU(True),
                nn.Linear(36, 54),
                nn.ReLU(True),
                nn.Linear(54, 18)
            )
        self.classifier = nn.Sequential(
                nn.Linear(16, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 32),
                nn.Linear(32, 18)
            )


    def forward(self, x):
        # x = self.conv(x).view(-1, 18)
        # return self.fc(x).view(-1, 18)
        # return self.classifier(x.view(-1, 16))
        return self.conv(x).view(-1, 18)


class ShallowNet(nn.Module):
    """docstring for ShallowNet"""
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(2, 6, 5),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3),
                nn.Conv2d(6, 6, 4),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3),
                nn.Conv2d(6, 6, 4),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3),
                nn.Conv2d(6, 12, 5),
                nn.Conv2d(12, 18, 4)
            )


    def forward(self, x):
        return self.classifier(x).view(-1, 18)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0, std=10)
        nn.init.normal_(m.bias.data, mean=0, std=10)


net = Justreducelr_0()
net.apply(weights_init)

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
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.1)
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100,
            threshold=1e-6, factor=0.5, min_lr=1e-6)
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
            if i % record_cnt == record_cnt - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / record_cnt))
                with open(record_file, 'a') as f:
                    f.write(str(running_loss / record_cnt) + '\n')
                running_loss = 0.0
                torch.save(net.state_dict(), model_path)

    print('Finished Training')
    torch.save(net.state_dict(), model_path)
    return net


def test(net, net_name, dataset=test_set):
    correct = 0
    total = 0
    loss = 0
    sigmoid = nn.Sigmoid()
    # threshold = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, shuffle=True)
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = sigmoid(outputs)
            one_correct = 0
            for i in np.arange(0.3, 0.7, 0.01):
                tmp_outputs = outputs.clone()
                tmp_outputs[tmp_outputs > i] = 1
                tmp_outputs[tmp_outputs <= i] = 0
                tmp_correct = (tmp_outputs.data == labels).sum().item()
                if tmp_correct > one_correct:
                    one_correct = tmp_correct
                    # threshold = i
                    r_outputs = tmp_outputs.clone()
            # print('--------threshold', threshold)
            outputs = r_outputs
            # outputs = torch.round(outputs)
            total += labels.size(0)*label_len
            correct += one_correct
            tmp_loss = abs(outputs.data - labels).sum().item()
            loss += tmp_loss

    print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total))
    print('Loss of the network: {}'.format(loss))
    with open(record_file, 'a') as f:
        f.write('This is the result of' + net_name + '\n')
        f.write(str(100 * correct / total) + '\n')


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
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 12, 3)
        self.conv4 = nn.Conv2d(12, 15, 2)
        self.conv5 = nn.Conv2d(15, 18, 3)
        self.norm1 = nn.GroupNorm(3, 6)
        self.norm2 = nn.GroupNorm(4, 12)
        self.norm3 = nn.GroupNorm(5, 15)
        self.norm4 = nn.GroupNorm(6, 18)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        first = torch.unsqueeze(x[:, 0], 1)
        second = torch.unsqueeze(x[:, 1], 1)
        # third = x[:, 2]
        first = (F.relu(self.norm1(self.conv1(first))))
        # first = self.pool(F.relu(self.norm1(self.conv2(first))))
        for i in range(124):
            first = F.relu(self.norm1(self.conv2(first)))
        first = F.relu(self.norm2(self.conv3(first)))
        first = F.relu(self.norm3(self.conv4(first)))
        first = F.relu(self.norm4(self.conv5(first)))

        second = (F.relu(self.norm1(self.conv1(second))))
        # second = self.pool(F.relu(self.norm1(self.conv2(second))))
        for i in range(124):
            second = F.relu(self.norm1(self.conv2(second)))
        second = F.relu(self.norm2(self.conv3(second)))
        second = F.relu(self.norm3(self.conv4(second)))
        second = F.relu(self.norm4(self.conv5(second)))
        y = (first + second).view(-1, 18)
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
