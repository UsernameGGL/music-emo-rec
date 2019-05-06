import torch
from train_test import Coon_0_2 as Net
from train_test import test, set_record_file
from MusicDataset import MusicDataFour, IniNorm
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_dir = basic_dir + 'raw-data-v5/'
label_file = basic_dir + 'labels-v5.csv'
net_name = '2-0+normcoon'
model_name = net_name + '.pt'
set_record_file(net_name + '.txt')
net = Net()
net.load_state_dict(torch.load(model_name))
net.eval()
tsfm = IniNorm()
test_set = MusicDataFour(data_dir=data_dir, label_file=label_file, transform=tsfm)
test(net, net_name=net_name, dataset=test_set)

