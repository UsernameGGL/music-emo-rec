from train_test import train, test
from train_test import Norm_0_1 as Net
# time.sleep(3600*4)

print('start')
net_name = '1-0+norm'
model_path = net_name + '.pt'
net = Net()
net = train(net, model_path=model_path)
test(net, net_name)


