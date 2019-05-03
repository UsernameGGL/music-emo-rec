from train_test import train, test
from train_test import Coon_0_2 as Net

# time.sleep(3600*4)

print('start')
model_path = '2-0+coon.pt'


if __name__ == "__main__":
    net = Net()
    net = train(net, model_path)
    test(net)


