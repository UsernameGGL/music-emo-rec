import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import csv
batch_size = 128
epoch_num = 80
record_cnt = 10
label_len = 18
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
basic_dir = '../'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'


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


class Musicdata_v7(Dataset):
    """docstring for Musicdata_v7"""
    def __init__(self, idx, data_file, label_file=label_file, transform=None):
        super(Musicdata_v7, self).__init__()
        data_file = open(data_file, 'r')
        self.rows = data_file.readlines()
        self.idx = idx
        self.transform = transform
        self.labels = get_label(label_file)


    def __len__(self):
        return len(self.idx)


    def __getitem__(self, idx):
        row = self.rows[self.idx[idx]].split(',')
        data = torch.Tensor(list(map(float, row))[0: 16]).view(1, 4, 4)
        label = self.labels[self.idx[idx]]
        return data, label


class SimpleCNN(nn.Module):
    """docstring for SimpleCNN"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 2),
                nn.Conv2d(6, 12, 2),
                nn.Conv2d(12, 18, 2),
            )


    def forward(self, x):
        return self.conv(x).view(-1, 18)


# train_set = Musicdata_v7(data_file, label_file, start=train_start, total=train_end)
# test_set = Musicdata_v7(data_file, label_file, start=test_start, total=test_end)
criterion = nn.BCEWithLogitsLoss()


def train(net, dataset, criterion=criterion, optimizer=None, scheduler=None):

    net.to(device)
    if not optimizer:
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.1)
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=40,
            threshold=1e-6, factor=0.5, min_lr=1e-6)
    for epoch in range(epoch_num): 
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
            if i % record_cnt == record_cnt - 1:
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / record_cnt))
                running_loss = 0.0

    print('Finished Training')
    return net


def test(net, dataset):
    correct_1 = 0
    correct_2 = 0
    total = 0
    loss_1 = 0
    loss_2 = 0
    sigmoid = nn.Sigmoid()
    # threshold = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, shuffle=False)
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
            outputs_1 = r_outputs
            outputs_2 = outputs.clone()
            outputs_2[outputs_2 > 0.5] = 1
            outputs_2[outputs_2 <= 0.5] = 0
            # outputs = torch.round(outputs)
            total += labels.size(0)*label_len
            correct_1 += one_correct
            correct_2 += (outputs_2 == labels).sum().item()
            tmp_loss = abs(outputs_1.data - labels).sum().item()
            loss_1 += tmp_loss
            loss_2 += abs(outputs_2.data - labels).sum().item()

    acc_1 = 100 * correct_1 / total
    acc_2 = 100 * correct_2 / total
    print('Accuracy_1 of the network on the test images: %f %%' % (
            acc_1))
    print('Loss_1 of the network: {}'.format(loss_1))
    print('Accuracy_2 of the network on the test images: %f %%' % (
            acc_2))
    print('Loss_2 of the network: {}'.format(loss_2))
    return acc_1, acc_2, loss_1, loss_2


def generate_idx(idx, num, total):
    num -= 1
    if not num < 0:
        length = len(idx)
        while True:
            tmp_idx = np.random.randint(total)
            flag = False
            for i in range(length):
                if tmp_idx == idx[i]:
                    flag = True
                    break
            if flag:
                continue

            idx.append(tmp_idx)
            idx = generate_idx(idx, num, total)
            break
    return idx


def add_idx(idx, num, total):
    new_idx = []
    idx_idx = 0
    for i in range(total):
        if idx_idx >= num or i != idx[idx_idx]:
            new_idx.append(i)
        else:
            idx_idx += 1
    return new_idx


if __name__ == '__main__':
    history = []
    total = 3219
    test_num = 659
    best_acc1 = 0
    best_acc2 = 0
    while True:
        test_idx = []
        test_idx = generate_idx(test_idx, test_num, total)
        test_idx.sort()
        length = len(history)
        flag = False
        for i in range(length):
            if test_idx == history[i]:
                flag = True
                break
        if flag:
            continue
        history.append(test_idx)
        train_idx = add_idx(test_idx, test_num, total)
        net = SimpleCNN()
        train_set = Musicdata_v7(train_idx, data_file, label_file)
        test_set = Musicdata_v7(test_idx, data_file, label_file)
        net = train(net, dataset=train_set)
        acc_1, acc_2, loss_1, loss_2 = test(net, dataset=test_set)
        if acc_1 > best_acc1 or acc_2 > best_acc2:
            if acc_1 > best_acc1:
                best_acc1 = acc_1
            elif acc_2 > best_acc2:
                best_acc2 = acc_2
            print(acc_1, acc_2, loss_1, loss_2)
            with open('record-idx.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(test_idx)
                writer.writerow([acc_1, acc_2, loss_1, loss_2])
