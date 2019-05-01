import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import csv
import time
import _thread
import statistics
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchvision import transforms
# from torchvision.utils import save_image

# time.sleep(3600*18)

# train_rate = 0.8
train_slice_num = 2223  # 用来训练的曲子数
total_slice_num = train_slice_num + 1000
pic_len = 256           # 图片长度
batch_size = 100
epoch_num = 1
# input_num = 2
interval = 2000            # 窗口间隔
part_data_num = 1000     # 每一次训读入的数据
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

print('start')



# print('slice_num is {}'.format(len(input_slice)))
# print('music input end, start to label')
# with open('record', 'a') as f:
#     f.write('slice_num is {}\nmusic input end, start to label\n'.format(slice_num))

# 读入label
label_file = open('../../data/cal500/prodLabels.csv', 'r')
label_reader = csv.reader(label_file)
#print(list(label_reader)[0:5])
reader_list = list(label_reader)
create_labels = True
for line in reader_list:
    if line == []:
        continue
    if create_labels:
        input_label = [list(map(int, line))]
        create_labels = False
    else:
        input_label.append(list(map(int, line)))
label_len = len(input_label[0])
label_file.close()
#labels = list(map(int, label_reader))[0:20]
# print(labels[input_num-1])


# 对每一行取出多个128*128数据执行傅里叶变换，把时频数据分别放入不同的channel，就得到了一张图片，多次循环得到多张图片
# train_num = math.ceil(slice_num*train_rate)
# create_train_tensor = True
# create_test_tensor = True
train_data = torch.Tensor(part_data_num, 2, pic_len, pic_len)
test_data = torch.Tensor(part_data_num, 2, pic_len, pic_len)
train_label = torch.Tensor(part_data_num, label_len)
test_label = torch.Tensor(part_data_num, label_len)
train_or_test = 'train'
true_data_len = 0
train_data_is_left = True
test_data_is_left = True
train_data_has_refreshed = False
test_data_has_refreshed = False
slice_num = 0


def transfer_data():
    # global slice_num
    # global input_slice
    global pic_len
    global train_or_test
    global train_data
    global test_data
    global interval
    global true_data_len
    global part_data_num
    global train_slice_num
    global train_data_is_left
    global test_data_is_left
    global train_data_has_refreshed
    global test_data_has_refreshed
    global slice_num
    global total_slice_num
    # 读入音频数据，计算数据行数
    input_file = open('../../data/cal500/prodAudios_v2.txt', 'r')
    input_lines = input_file.readlines()
    # create_inp_list = True
    for line in input_lines:
        slice_num += 1
        line = line.split(' ')
        line.pop()
        line = list(map(int, line))
        line_len = len(line)
        # print(line_len)
        # print(line_len)
        sample_start = 0
        sample_len = pic_len*pic_len
        while sample_start + sample_len <= line_len:
            time_data = line[sample_start: sample_start + sample_len]
            freq_data = abs(np.fft.fft(time_data)/sample_len)
            # freq_data[0] = statistics.mean(freq_data[1:])
            # #######################################here
            time_data = np.array(time_data).reshape(pic_len, pic_len)
            freq_data = np.array(freq_data).reshape(pic_len, pic_len)
            sample_start += interval
            if train_or_test == 'train':
                train_data[true_data_len] = torch.Tensor([time_data, freq_data])
                train_label[true_data_len] = torch.Tensor(input_label[slice_num - 1])
            else:
                test_data[true_data_len] = torch.Tensor([time_data, freq_data])
                test_label[true_data_len] = torch.Tensor(input_label[slice_num - 1])
            true_data_len += 1
            if true_data_len == part_data_num:
                if train_or_test == 'train':
                    train_data_has_refreshed = True
                else:
                    test_data_has_refreshed = True
            while true_data_len == part_data_num:
                time.sleep(1)
        if slice_num >= train_slice_num:
            train_or_test = 'test'
            train_data_is_left = False
        if slice_num >= total_slice_num:
            break
    input_file.close()
    test_data_is_left = False



# 重写Dataset类
class TimeFreqDataset(Dataset):

    def __init__(self, data, label):
        self.len = len(data)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len
# 创建data_loader




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.conv5 = nn.Conv2d(6, 6, 3)
        self.norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 7)
        self.conv4 = nn.Conv2d(16, 8, 5)
        self.norm3 = nn.BatchNorm2d(8)
        linear_len = int(((pic_len - 4) / 2 - 2*10 - 4) / 2 - 6 - 4)
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


net = Net()
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
net.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()  # ##################################################### here
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min')


def train():
    global epoch_num
    global optimizer
    global net
    global criterion
    global train_data
    global train_label
    global batch_size
    global train_data_has_refreshed
    global true_data_len
    global part_data_num
    global device
    global slice_num
    global train_slice_num
    last_loss = 0
    loss_state_cnt = 0
    train_is_needed_cnt = 0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        while train_data_is_left:
            if not train_data_has_refreshed:
                time.sleep(0.5)
                continue
            if true_data_len < part_data_num:
                batch_size = 1
            train_data_has_refreshed = False
            true_data_len = 0
            tmp_train_data = train_data.clone()
            tmp_train_label = train_label.clone()
            train_loader = DataLoader(dataset=TimeFreqDataset(tmp_train_data, tmp_train_label),
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
                # print(outputs)
                # print(loss)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    loss_state_cnt += 1
                    running_loss = 0.0


    print('Finished Training')
    torch.save(net.state_dict(), '3 1+deeper.pt')


def test():
    global test_data
    global test_label
    global batch_size
    global test_data_is_left
    global net
    global test_data_has_refreshed
    global true_data_len
    global part_data_num
    global device
    correct = 0
    correct_v2 = 0
    total = 0
    loss = 0
    sigmoid = nn.Sigmoid()
    my_total = 0
    correct_v3 = 0
    correct_v4 = 0
    total_v4 = 0
    with torch.no_grad():
        while test_data_is_left:
            if not test_data_has_refreshed:
                time.sleep(0.5)
                continue
            if true_data_len < part_data_num:
                batch_size = 1
            test_data_has_refreshed = False
            true_data_len = 0
            tmp_test_data = test_data.clone()
            tmp_test_label = test_label.clone()
            test_loader = DataLoader(dataset=TimeFreqDataset(tmp_test_data, tmp_test_label),
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
                outputs = torch.round(outputs)
                # print(3, outputs)
                #labels = labels.float()
                total += labels.size(0)
                ########################################
                my_total += labels[labels == 1].sum().item()
                ########################################
                outputs[outputs < 0] = 0
                outputs[outputs > 1] = 1
                # print(4, outputs)
                ########################################
                correct += (outputs.data == labels).sum().item()
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
    _thread.start_new_thread(transfer_data, ())
    train()
    test()


