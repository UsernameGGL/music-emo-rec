import matplotlib.pyplot as plt
import numpy as np

record_files = ['record-shallownet.txt', 'record-fzdata.txt',
		'record-fzdata-fully.txt', 'record-fzdata-comb.txt']
labels = ['shallow convolutional network', 'fuzzy data with convolutional network',
		'fully connective network', 'network with both two layers']
# labels = ['d', 's', 'f', 'fu', 'comb']
idx = 0
plot_len = 10
for record_file in record_files:
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
	print(len(data))
	data = data[0: plot_len]
	epoch = [i for i in range(1, plot_len + 1)]
	plt.plot(epoch, data, label=labels[idx])
	idx += 1
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(0.55, 0.70)
plt.title('Speed of loss descending')
plt.show()
