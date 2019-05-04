from train_test import train, test
from train_test import Deeper_4_5 as Net

# time.sleep(3600*4)

print('start')
net_name = '5-4+deeper'
model_path = net_name + '.pt'
net = Net()
net = train(net, model_path=model_path)
test(net, net_name)


