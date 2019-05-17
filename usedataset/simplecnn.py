from train_test import SimpleCNN as Net
from train_test import train, test, set_record_file
from MusicDataset import Musicdata_v7
import sys
arg_len = len(sys.argv)
net_name = sys.argv[1]
# net_name = 'fzdata'
# net_name = 'fzdata-freq'
# net_name = 'fzdata-fully'
# net_name = 'fzdata-comb'
mode = sys.argv[2]
# mode = 'fully'
# mode = 'conv'
# mode = anyone else
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
basic_dir = '../'
if net_name == 'fzdata-freq':
	data_file = basic_dir + 'music-data-v9-freq.csv'
else:
	data_file = basic_dir + 'music-data-v9.csv'
# data_file = basic_dir + 'music-data-v7.csv'
# data_file = basic_dir + 'music-data-v9-freq.csv'

label_file = basic_dir + 'labels-v5.csv'
record_file = 'record-' + net_name + '.txt'
model_path = net_name + '.pt'
set_record_file(record_file)
net = Net(mode)
# net.apply(weights_init)
train_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=0, total=2560)
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
net = train(net, model_path=model_path, dataset=train_set)
test(net, net_name, dataset=test_set)
