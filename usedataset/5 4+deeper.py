from train_test import train, test
from train_test import Deeper_4_5 as Net

# time.sleep(3600*4)

print('start')
model_path = '5-4+deeper.pt'


if __name__ == "__main__":
    net = Net()
    net = train(net, model_path)
    test(net)


