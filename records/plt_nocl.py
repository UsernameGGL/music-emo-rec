import matplotlib.pyplot as plt
import numpy as np
def plt_nocl(record_file, net_name):
	file = open(record_file)
	data = []
	data_len = 0
	contents = file.readlines()
	for content in contents:
		length = len(content)
		for i in range(length):
			if i < length - 1 and content[i] == '0' and content[i+1] == '.':
				data.append(0)
				mul = 10
				for j in range(i+2, length):
					if j < length - 1 and content[j+1] != '.':
						data[data_len] += int(content[j])/mul
						mul *= 10
					else:
						i = j
						break
				data_len += 1
	data = data[0: 250]
	std_arr = np.array(data[50: ])
	print(np.std(std_arr))
	print(min(data))
	# print(len(data))
	length = len(data)
	epoch = [i for i in range(1, length + 1)]
	plt.plot(epoch, data)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	# plt.yticks([])
	# plt.ylim(0.55, 0.70)
	plt.title('Loss of ' + net_name)
	plt.show()
