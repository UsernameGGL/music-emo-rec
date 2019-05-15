from train_test import ShallowNet as Net
from train_test import train, test, set_record_file

print('start')
record_file = 'record-shallownet.txt'
model_path = 'shallownet.pt'
net_name = 'shallownet'
set_record_file(record_file)
net = Net()
net = train(net, model_path=model_path)
test(net, net_name)
