from train_test import train, test
from train_test import Deeper_1_3 as Net

# time.sleep(3600*4)

print('start')
model_path = '3-1+deeper.pt'


if __name__ == "__main__":
    net = Net()
    net = train(net, model_path)
    test(net)


