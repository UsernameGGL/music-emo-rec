from train_test import train, test
from train_test import Coon_1_4 as Net

# time.sleep(3600*4)

print('start')
model_path = '4-1+coon.pt'


if __name__ == "__main__":
    net = Net()
    net = train(net, model_path)
    test(net)


