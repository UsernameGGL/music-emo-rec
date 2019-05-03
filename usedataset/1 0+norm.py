from train_test import train, test
from train_test import Norm_0_1 as Net
# time.sleep(3600*4)

print('start')
model_path = '1-0+norm.pt'



if __name__ == "__main__":
	net = Net()
	net = train(net, model_path)
	test(net)


