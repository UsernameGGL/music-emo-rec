from train_test import train, test, set_record_file
from MusicDataset import MusicDataThree, IniNorm


print('start')
net_name = '0-justreducelr-ininorm'
model_path = net_name + '.pt'
set_record_file('record-normreducelr.txt')
tsfm = IniNorm()
train_set = MusicDataThree(transform=tsfm, start=0, total=1)
test_set = MusicDataThree(transform=tsfm, start=0, total=1)
net = train(model_path=model_path, dataset=train_set)
test(net, net_name, dataset=test_set)


