from train_test import train, test

# time.sleep(3600*16)


print('start')


if __name__ == "__main__":
    net = train(model_path='0-justreducelr.pt')
    test(net)


