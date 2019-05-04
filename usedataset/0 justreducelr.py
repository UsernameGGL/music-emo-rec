from train_test import train, test

# time.sleep(3600*16)


print('start')
net_name = '0-justreducelr'
model_path = net_name + '.pt'
net = train(model_path=model_path)
test(net, net_name)


