import torch
import torch.nn as nn
import numpy as np
# import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
criterion = nn.BCEWithLogitsLoss()
epoch_num = 1000
device = torch.device('cpu')
batch_size = 128
label_len = 18


def train(net, criterion=criterion, model_path='tmp.pt', dataset=None, optimizer=None, scheduler=None, record_file=None):
    with open(record_file, 'a') as f:
        f.write('This is model of' + model_path + '\n')

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
            h0 = torch.randn(2, 1, 18)
            c0 = torch.randn(2, 1, 18)
            outputs, _ = net(inputs, (h0, c0))
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


def test(net, net_name, dataset=None, record_file=None):
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
            outputs, _ = net(images)
            # print(1, outputs)
            # ####################################################here
            outputs = sigmoid(outputs).view(-1, 18)
            labels = labels.view(-1, 18)
            # print(2, outputs)
            for k in range(len(outputs)):
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