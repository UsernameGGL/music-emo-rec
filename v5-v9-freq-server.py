import csv
import numpy as np
basic_dir = '../'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
v5 = list(csv.reader(open(data_file, 'r')))
sample_len = 256*256
length = len(v5)
v9 = [[] for i in range(length)]
max_list = [0 for i in range(length)]
for i in range(length):
	v5[i] = list(map(float, v5[i]))
	v9[i] = abs(np.fft.fft(v5[i][0: sample_len]) / sample_len)
	v9[i][0] = v9[i][1]
	max_list[i] = max(v9[i])
max_data = max(max_list)
print('max_data is {}'.format(max_data))

out_list = [[] for i in range(length)]
idx = 0
sum_num = 4096
need_num = 16
for i in range(length):
	out = []
	for j in range(need_num):
		out.append(sum(v9[i][j: j + sum_num]) / sum_num)
	out_list[idx] = out
	idx += 1
max_data = max(max(out_list))

out_file = open(basic_dir + 'music-data-v9-freq.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for i in range(length):
	for j in range(need_num):
		out_list[i][j] = out_list[i][j] / max_data
	writer.writerow(out_list[i])	
out_file.close()


