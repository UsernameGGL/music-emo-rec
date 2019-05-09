import torch
import torch.nn as nn
from train_test import test, set_record_file
from MusicDataset import Musicdata_LSTM
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
# data_dir = basic_dir + 'raw-data-v5/'
data_file = basic_dir + 'music-data-v7.csv'
label_file = basic_dir + 'labels-v5.csv'
net_name = 'lstm'
model_name = net_name + '.pt'
set_record_file(net_name + '.txt')
net = nn.LSTM(16, 18, 2)
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=3219)
test(net, net_name=net_name, dataset=test_set)

