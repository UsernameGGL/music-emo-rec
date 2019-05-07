from train_test import train, test, set_record_file

# time.sleep(3600*16)


print('start')
net_name = '0-justreducelr'
model_path = net_name + '.pt'
set_record_file('record' + net_name + '.txt')
net = train(model_path=model_path)
test(net, net_name)


