from train_test import train, test
from train_test import Ininorm_5_6 as Net
from MusicDataset import MusicDataThree, IniNorm

# time.sleep(3600*4)

print('start')
model_path = '6-5+ininorm.pt'


print('start')
model_path = '6-5+ininorm.pt'
tsfm = IniNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)
net = Net()
net = train(net, model_path=model_path, dataset=train_set)
test(net, dataset=test_set)




