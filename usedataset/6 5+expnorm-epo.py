from train_test import train, test
from train_test import Expnorm_5_6 as Net
from MusicDataset import MusicDataThree, ExpNorm
import numpy as np
import torch

# time.sleep(3600*4)

print('start')
model_path = '6-5+expnorm.pt'


tsfm = ExpNorm()
train_set = MusicDataThree(transform=tsfm)
test_set = MusicDataThree(transform=tsfm)

if __name__ == "__main__":
    net = Net()
    net = train(net, model_path=model_path, dataset=train_set)
    test(net, dataset=test_set)




