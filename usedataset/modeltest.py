import torch
from train_test import SimpleCNN as Net
from train_test import test, set_record_file
from MusicDataset import MusicDataThree
from MusicDataset import Musicdata_v7
# from MusicDataset import Musicdata_LSTM
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
basic_dir = '../'
# data_dir = basic_dir + 'raw-data-v5/'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
# net_name = 'shallownet'
net_name = 'simplecnn-time'
# net_name = 'simplecnn-freq'
model_name = net_name + '.pt'
set_record_file('record-' + net_name + '.txt')
net = Net()
net.load_state_dict(torch.load(model_name))
net.eval()
# test_set = MusicDataThree(data_file=data_file, label_file=label_file, start=2560, total=3219)
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
test(net, net_name=net_name, dataset=test_set)


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

