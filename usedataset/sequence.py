from train_test import train, test
from train_test import Norm_0_1, Coon_0_2, Deeper_1_3, Coon_1_4
from train_test import Deeper_4_5
from train_test import Expnorm_5_6
from MusicDataset import MusicDataThree, ExpNorm
from train_test import Ininorm_5_6
from MusicDataset import IniNorm
from train_test import trans_mode

# time.sleep(3600*16)


# print('start')
# net_name = '0-justreducelr'
# model_path = net_name + '.pt'
# net = train(model_path=model_path)
# test(net, net_name)

# print('start')
# net_name = '1-0+norm'
# model_path = net_name + '.pt'
# net = Norm_0_1()
# net = train(net, model_path=model_path)
# test(net, net_name)

# print('start')
# net_name = '2-0+coon'
# model_path = net_name + '.pt'
# net = Coon_0_2()
# net = train(net, model_path=model_path)
# test(net, net_name)

# print('start')
# net_name = '3-1+deeper'
# model_path = net_name + '.pt'
# net = Deeper_1_3()
# net = train(net, model_path=model_path)
# test(net, net_name)

# print('start')
# net_name = '4-1+coon'
# model_path = net_name + '.pt'
# net = Coon_1_4()
# net = train(net, model_path=model_path)
# test(net, net_name)

# print('start')
# net_name = '5-4+deeper'
# model_path = net_name + '.pt'
# net = Deeper_4_5()
# net = train(net, model_path=model_path)
# test(net, net_name)

print('start')
model_path = '6-5+expnorm.pt'
tsfm = ExpNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)
net = Expnorm_5_6()
net = train(net, model_path=model_path, dataset=train_set)
test(net, dataset=test_set)

print('start')
model_path = '6-5+ininorm.pt'
tsfm = IniNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)
net = Ininorm_5_6()
net = train(net, model_path=model_path, dataset=train_set)
test(net, dataset=test_set)


trans_mode('one-hot')


print('start')
net_name = '0-justreducelr'
model_path = net_name + '.pt'
net = train(model_path=model_path)
test(net, net_name)

print('start')
net_name = '1-0+norm'
model_path = net_name + '.pt'
net = Norm_0_1()
net = train(net, model_path=model_path)
test(net, net_name)

print('start')
net_name = '2-0+coon'
model_path = net_name + '.pt'
net = Coon_0_2()
net = train(net, model_path=model_path)
test(net, net_name)

print('start')
net_name = '3-1+deeper'
model_path = net_name + '.pt'
net = Deeper_1_3()
net = train(net, model_path=model_path)
test(net, net_name)

print('start')
net_name = '4-1+coon'
model_path = net_name + '.pt'
net = Coon_1_4()
net = train(net, model_path=model_path)
test(net, net_name)

print('start')
net_name = '5-4+deeper'
model_path = net_name + '.pt'
net = Deeper_4_5()
net = train(net, model_path=model_path)
test(net, net_name)

print('start')
model_path = '6-5+expnorm.pt'
tsfm = ExpNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)
net = Expnorm_5_6()
net = train(net, model_path=model_path, dataset=train_set)
test(net, dataset=test_set)

print('start')
model_path = '6-5+ininorm.pt'
tsfm = IniNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)
net = Ininorm_5_6()
net = train(net, model_path=model_path, dataset=train_set)
test(net, dataset=test_set)
