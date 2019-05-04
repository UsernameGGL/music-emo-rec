import torch
from train_test import Justreducelr_0 as Net
from train_test import test, set_record_file
from MusicDataset import MusicDataFour
net_name = '0-justreducelr'
model_name = net_name + '.pt'
set_record_file(net_name + '.txt')
net = Net()
net.load_state_dict(torch.load(model_name))
net.eval()
test_set = MusicDataFour()
test(net, net_name=net_name, dataset=test_set)

