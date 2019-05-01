import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import csv
import time
import _thread
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torchvision import transforms
# from torchvision.utils import save_image

time.sleep(3600*8)

# train_rate = 0.8
train_slice_num = 2223  # 用来训练的曲子数
total_slice_num = train_slice_num + 1000
pic_len = 256  # 图片长度
batch_size = 100
epoch_num = 1
# input_num = 2
interval = 2000  # 窗口间隔
part_data_num = 1000  # 每一次训读入的数据
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
# print(list(label_reader)[0:5])
reader_list = list(label_reader)
create_labels = True
for line in reader_list:
    if line == []:
        continue
    if create_labels:
        labels = [list(map(int, line))]
        create_labels = False
    else:
        labels.append(list(map(int, line)))
label_len = len(labels[0])
label_file.close()
# labels = list(map(int, label_reader))[0:20]
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
    path = 'E:/data/cal500/music-data-v2/'
    sliceDir = os.listdir(path=path)
    sample_len = pic_len*pic_len
    for sliceFNama in sliceDir:
        slice_file = open(path + sliceFNama, 'r')
        slice_reader = csv.reader(slice_file)
        slices = list(slice_reader)
        slice_num += 1
        for one_slice in slices:
            if not one_slice:
                continue
            one_slice = list(map(float, one_slice))
            time_data = one_slice[0: sample_len]
            freq_data = one_slice[sample_len: len(one_slice)-1]
            time_data = np.array(time_data).reshape(pic_len, pic_len)
            freq_data = np.array(freq_data).reshape(pic_len, pic_len)
            if train_or_test == 'train':
                train_data[true_data_len] = torch.Tensor([time_data, freq_data])
                train_label[true_data_len] = torch.Tensor(labels[slice_num - 1])
            else:
                test_data[true_data_len] = torch.Tensor([time_data, freq_data])
                test_label[true_data_len] = torch.Tensor(labels[slice_num - 1])
            true_data_len += 1
            if true_data_len == part_data_num:
                if train_or_test == 'train':
                    train_data_has_refreshed = True
                else:
                    test_data_has_refreshed = True
            while true_data_len == part_data_num:
                time.sleep(0.5)
        if train_data_is_left and slice_num >= train_slice_num:
            train_or_test = 'test'
            train_data_is_left = False
        if slice_num >= total_slice_num:
            break
        slice_file.close()
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
        y = F.softmax(first + second)
        return y


net = Net()
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
net.to(device)

# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()  # ##################################################### here
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
    # last_loss = 0
    # loss_state_cnt = 0
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
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    # with open('record', 'a') as f:
                    #     f.write('[%d, %5d] loss: %.3f\n\n' %
                    #         (epoch + 1, i + 1, running_loss / 2000))
                    # if last_loss <= running_loss:
                    #     loss_state_cnt += 1
                    #     if loss_state_cnt >= 10:
                    #         optimizer.param_groups[0]['lr'] *= 0.1
                    #         loss_state_cnt = 0
                    # last_loss = running_loss
                    running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), '2 0+coon-sof.pt')


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
    cnt = 0
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
                # labels = labels.float()
                total += labels.size(0)
                #########################################################
                my_total += labels[labels == 1].sum().item()
                ########################################
                # outputs[outputs < 0] = 0
                # outputs[outputs > 1] = 1
                ########################################
                correct += (outputs.data == labels).sum().item()
                # loss += abs(outputs.data - labels).sum().item()
                tmp_loss = abs(outputs.data - labels).sum().item()
                if tmp_loss == 0:
                    correct_v2 += 1
                loss += tmp_loss
                # label_batch = labels.size(0)
                # batch_len = labels.size(1)
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
