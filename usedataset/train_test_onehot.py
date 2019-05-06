import torch
import torch.nn as nn
# import sys
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from MusicDataset import MusicDataThree
from torch.optim.lr_scheduler import ReduceLROnPlateau
train_start = 0
train_end = 2578  # 用来训练的曲子数
test_start = 2578
test_end = 3219
batch_size = 100
pic_len = 256
label_len = 18
epoch_num = 3
type_num = 855
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'one-hot-label.csv'
record_file = 'record-one-hot.txt'

def set_record_file(file_name):
    global record_file
    record_file = file_name

class Coon_0_2(nn.Module):
    def __init__(self):
        super(Coon_0_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 12, 3)
        self.conv4 = nn.Conv2d(12, 6, 3)
        # self.conv5 = nn.Conv2d(6, 6, 5)
        # self.conv6 = nn.Conv2d(6, 1, 2)
        self.fc = nn.Linear(1176, type_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        first = torch.unsqueeze(x[:, 0], 1)
        second = torch.unsqueeze(x[:, 1], 1)
        # third = x[:, 2]
        first = F.relu(self.pool(self.conv1(first)))
        first = F.relu(self.pool(self.conv2(first)))
        first = F.relu(self.pool(self.conv3(first)))
        first = F.relu(self.pool(self.conv4(first)))
        # for i in range(3):
        #     first = F.relu(self.conv5(first))
        # first = self.conv6(first)
        first = first.view(batch_size, -1)

        second = F.relu(self.pool(self.conv1(second)))
        second = F.relu(self.pool(self.conv2(second)))
        second = F.relu(self.pool(self.conv3(second)))
        second = F.relu(self.pool(self.conv4(second)))
        # for i in range(3):
        #     second = F.relu(self.conv5(second))
        # second = self.conv6(second)
        second = second.view(batch_size, -1)
        y = self.sigmoid(first + second)
        return y


net = Coon_0_2()

# criterion = nn.BCELoss()  # ##################################################### here
train_set = MusicDataThree(data_file, label_file, start=train_start, total=train_end, mode='one-hot')
test_set = MusicDataThree(data_file, label_file, start=test_start, total=test_end, mode='one-hot')
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()


def train(net=net, criterion=criterion, model_path='tmp.pt', dataset=train_set, optimizer=None, scheduler=None):
    with open(record_file, 'a') as f:
        f.write('This is model of' + model_path + '\n')

    net.to(device)
    if not optimizer:
        optimizer = torch.optim.SGD(net.parameters(), lr=10, momentum=0.1)
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30, 
            threshold=1e-6, factor=0.7, min_lr=1e-7)
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
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                with open(record_file, 'a') as f:
                    f.write('[%d, %5d] loss: %.3f\n' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                torch.save(net.state_dict(), model_path)

    print('Finished Training')
    torch.save(net.state_dict(), model_path)
    return net


def test(net, net_name, dataset):
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, shuffle=True)
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            total += batch_size
            for k in range(batch_size):
                _, idx = torch.sort(outputs[k], descending=True)
                if idx[0] == labels[k]:
                    correct += 1
                else:
                    loss += (idx[0] - labels[k]) * (idx[0] - labels[k])
            # tmp_loss = abs(outputs.data - labels).sum().item()
            # loss += tmp_loss
            # print('Accuracy of the network on the test images: %f %%' % (
            # 100 * correct / total))
            # print('Loss of the network: {}'.format(loss))
            print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total))
            print('Loss of the network: {}'.format(loss))

    print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total))
    print('Loss of the network: {}'.format(loss))
    with open(record_file, 'a') as f:
        f.write('This is the result of' + net_name + '\n')
        f.write('Accuracy of the network on the test images: %f %%\n' % (
            100 * correct / total))
        f.write('Loss of the network: {}\n'.format(loss))

