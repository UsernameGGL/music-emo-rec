from train_test import train, test, set_record_file, Justreducelr_0_ini

# time.sleep(3600*16)


print('start')
net_name = '0-justreducelr'
model_path = net_name + '.pt'
set_record_file('record' + net_name +'.txt')
net = Justreducelr_0_ini()
net = train(net=net, model_path=model_path)
test(net, net_name)


