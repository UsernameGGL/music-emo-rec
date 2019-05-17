import torch.nn as nn
from train_test_lstm import train, test
from MusicDataset import Musicdata_LSTM
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v7.csv'
label_file = basic_dir + 'labels-v5.csv'
record_file = 'record-lstm.txt'
model_path = 'lstm.pt'
net_name = 'lstm'
net = nn.LSTM(16, 18, 2)
train_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=2560)
test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=2560, total=3219)
net = train(net, model_path=model_path, dataset=train_set)
test(net, net_name, dataset=test_set)
