from train_test import SimpleCNN as Net
from train_test import train, test
from MusicDataset import Musicdata_v7
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v7.csv'
label_file = basic_dir + 'labels-v5.csv'
record_file = 'record-simplecnn.txt'
model_path = 'simplecnn.pt'
net_name = 'simplecnn'
net = Net()
# net.apply(weights_init)
train_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=0, total=2560)
test_set = Musicdata_v7(data_file=data_file, label_file=label_file, start=2560, total=3219)
net = train(net, model_path=model_path, dataset=train_set)
test(net, net_name, dataset=test_set)
