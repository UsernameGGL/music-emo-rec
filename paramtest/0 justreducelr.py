import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import _thread
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MusicDataset import MusicDataThree

time.sleep(3600*16)

# train_rate = 0.8
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
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
net.to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()  # ##################################################### here
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min')


def train():
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        train_loader = DataLoader(dataset=MusicDataThree(data_file, label_file),
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
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


    print('Finished Training')
    torch.save(net.state_dict(), '0 justreducelr.pt')


def test():
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


