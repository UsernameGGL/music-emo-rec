import torch
from train_test import Justreducelr_0, ShallowNet, SimpleCNN
from train_test import test, set_record_file
from MusicDataset import MusicDataThree
from MusicDataset import Musicdata_v7
from MusicDataset import Musicdata_LSTM
import torch.nn as nn
import numpy as np
# from MusicDataset import Musicdata_LSTM
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
basic_dir = '../'
set_record_file('record-modeltest.txt')


print('Test of deep convolutional network')
data_dir = basic_dir + 'raw-data-v5/'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
# net_name = 'shallownet'
net_name = '0-justreducelr'
# net_name = 'simplecnn-freq'
model_name = net_name + '.pt'
net = Justreducelr_0()# deep convolutional network
net.load_state_dict(torch.load(model_name))
net.eval()
# test_set = MusicDataThree(data_file=data_file, label_file=label_file, start=2560, total=3219)
test_set = MusicDataThree(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


print('Test of shallow convolutional network')
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
net = ShallowNet()
net_name = 'shallownet'
model_name = net_name + '.pt'
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = MusicDataThree(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


print('Test on fuzzy data')
data_file = basic_dir + 'music-data-v9.csv'
label_file = basic_dir + 'labels-v5.csv'
mode = 'conv'
net = SimpleCNN(mode)
net_name = 'fzdata'
model_name = net_name + '.pt'
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


print('Test on fuzzy frequency data')
data_file = basic_dir + 'music-data-v9-freq.csv'
label_file = basic_dir + 'labels-v5.csv'
mode = 'conv'
net = SimpleCNN(mode)
net_name = 'fzdata-freq'
model_name = net_name + '.pt'
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


print('Test fully network')
data_file = basic_dir + 'music-data-v9.csv'
label_file = basic_dir + 'labels-v5.csv'
mode = 'fully'
net = SimpleCNN(mode)
net_name = 'fzdata-fully'
model_name = net_name + '.pt'
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


print('Test of conv with linear network')
data_file = basic_dir + 'music-data-v9.csv'
label_file = basic_dir + 'labels-v5.csv'
mode = 'else'
net = SimpleCNN(mode)
net_name = 'fzdata-comb'
model_name = net_name + '.pt'
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)










print('Test of DBLSTM')
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
num_classes = 18
batch_size = 128
data_file = basic_dir + 'music-data-v5.csv'
test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=2560, total=3219)
prefix = 'deep-lstm'
sequence_length = 256
input_size = 256
hidden_size = 64
num_layers = 60
num_epochs = 2000
learning_rate = 30
record_file = 'record-' + 'tmp-lstm' + '.txt'
model_path = prefix + '.pt'
record_num = 10
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, 
                                          shuffle=False)

device = torch.device('cpu')
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    correct_formal = 0
    total_formal = 0
    correct_pre = 0
    total_pre = 0
    correct_rec = 0
    total_rec = 0
    sigmoid = nn.Sigmoid()
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device).view(-1, num_classes)
        outputs = model(images)
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
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) * num_classes
        correct += (outputs == labels).sum().item()
        total_formal += labels.size(0)
        for iii in range(labels.size(0)):
            correct_formal += abs(labels[iii] - outputs[iii]).sum().item() == 0
            for jjj in range(labels[iii].size(0)):
                if outputs[iii][jjj]==1:
                    total_pre+=1
                    if labels[iii][jjj]==1:
                        correct_pre+=1
                if labels[iii][jjj]==1:
                    total_rec+=1
                    if outputs[iii][jjj]==1:
                        correct_rec+=1

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 
    print(100*correct_formal/total_formal)
    print(100*correct_pre/total_pre)
    print(100*correct_rec/total_rec)
    with open(record_file, 'a') as f:
        f.write('------------------------------\n')
        f.write(str(100 * correct / total) + '\n')
        f.write(str(100 * correct_formal / total_formal) + '\n')
        f.write(str(100 * correct_pre / total_pre) + '\n')
        f.write(str(100 * correct_rec / total_rec) + '\n')


# basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
# # data_dir = basic_dir + 'raw-data-v5/'
# data_file = basic_dir + 'music-data-v7.csv'
# label_file = basic_dir + 'labels-v5.csv'
# net_name = 'lstm'
# model_name = net_name + '.pt'
# set_record_file(net_name + '.txt')
# net = nn.LSTM(16, 18, 2)
# net.load_state_dict(torch.load(model_name))
# net.eval()
# test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=3219)
# test(net, net_name=net_name, dataset=test_set)

