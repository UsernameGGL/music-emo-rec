import torch
import torch.nn as nn
import numpy as np
# import sys
from torch.utils.data import DataLoader
from MusicDataset import Musicdata_Bin
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_start = 0
train_end = 2560  # 用来训练的曲子数
test_start = 2560
test_end = 3219
batch_size = 128
pic_len = 256
label_len = 18
epoch_num = 30
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v7.csv'
label_file = basic_dir + 'labels-v5.csv'
record_file = 'record-bin.txt'


class CNN_BIN(nn.Module):
    """docstring for SimpleCNN"""
    def __init__(self):
        super(CNN_BIN, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 2),
                nn.ReLU(True),
                nn.Conv2d(6, 6, 2),
                nn.ReLU(True),
                nn.Conv2d(6, 2, 2),
                nn.ReLU(True)
            )


        self.classifier = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(True),
                nn.Linear(32, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 32),
                nn.ReLU(True),
                nn.Linear(32, 2)
            )


    def forward(self, x):
        # x = self.conv(x).view(-1, 18)
        # return self.fc(x).view(-1, 18)
        return self.conv(x).view(-1, 2)


net = CNN_BIN()
criterion = nn.CrossEntropyLoss()


def train(net=net, criterion=criterion, model_path='bin.pt', clsf_idx=0, optimizer=None, scheduler=None):
    with open(record_file, 'a') as f:
        f.write('This is model of' + model_path + '\n')
    dataset = Musicdata_Bin(data_file, label_file, start=train_start, total=train_end, clsf_idx=clsf_idx)

    net.to(device)
    if not optimizer:
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.1)
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30, 
            threshold=1e-6, factor=0.5, min_lr=1e-6)
    running_loss = 0.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size, shuffle=True)
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
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / 10))
                with open(record_file, 'a') as f:
                    f.write('[%d, %5d] loss: %f\n' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                torch.save(net.state_dict(), model_path)

    print('Finished Training')
    torch.save(net.state_dict(), model_path)
    return net


def test(net, net_name, clsf_idx=0):
    correct = 0
    correct_v2 = 0
    total = 0
    loss = 0
    sigmoid = nn.Sigmoid()
    threshold = 0
    dataset = Musicdata_Bin(data_file, label_file, start=test_start, total=test_end, clsf_idx=0)
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
            print(torch.min(outputs))
            print(torch.max(outputs))
            one_correct = 0
            for i in np.arange(0.3, 0.7, 0.01):
                tmp_outputs = outputs.clone()
                tmp_outputs[tmp_outputs > i] = 1
                tmp_outputs[tmp_outputs <= i] = 0
                tmp_correct = 0
                for ii in range(len(labels)):
                    tmp_correct += outputs[ii][labels[ii]]
                if tmp_correct > one_correct:
                    one_correct = tmp_correct
                    threshold = i
                    r_outputs = tmp_outputs.clone()
            print('--------threshold', threshold)
            outputs = r_outputs
            # outputs = torch.round(outputs)
            total += labels.size(0)
            correct += one_correct
            print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total))

    print('Accuracy of the network on the test images: %f %%' % (
            100 * correct / total))
    with open(record_file, 'a') as f:
        f.write('This is the result of' + net_name + '\n')
        f.write('Accuracy of the network on the test images: %f %%\n' % (
            100 * correct / total))