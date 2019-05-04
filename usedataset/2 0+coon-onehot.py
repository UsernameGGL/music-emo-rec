from train_test import train, test, trans_mode, set_record_file
from train_test import Coon_0_2 as Net

# time.sleep(3600*4)

print('start')

trans_mode('one-hot')
set_record_file('record-coon-onehot.txt')
net_name = '2-0+coon-onehot'
model_path = net_name + '.pt'
net = Net()
net = train(net, model_path=model_path)
test(net, net_name)


