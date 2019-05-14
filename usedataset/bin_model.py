from train_test_bin import train, test
from train_test_bin import CNN_BIN
record_file = 'record-bin-cnn.txt'
model_path = 'bin-cnn.pt'
net_name = 'bin-cnn'
net = CNN_BIN()
net = train(net, model_path=model_path, clsf_idx=1)
test(net, net_name, clsf_idx=1)
