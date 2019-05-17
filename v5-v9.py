import csv
import os
sum_num = 4096
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_dir = basic_dir + 'raw-data-v5/'
# label_file = basic_dir + 'labels-v5.csv'
# data_file = basic_dir + 'music-data-v4.csv'
file_names = os.listdir(data_dir)
list_len = len(file_names)
out_list = [[] for j in range(list_len)]
idx = 0
for data_file in file_names:
	row = list(csv.reader(open(data_dir + data_file, 'r')))[0]
	row = list(map(int, row))
	length = len(row)
	row = list(map(int, row))
	i = 0
	out = []
	while i * sum_num + sum_num < length:
		out.append(sum(row[i * sum_num: i * sum_num + sum_num]) / sum_num)
		i += 1
	out_list[idx] = out
	idx += 1
	# writer.writerow(out)
print('加和完毕')
max_val = max(max(out_list))
data_file = basic_dir + 'music-data-v9.csv'
out_file = open(data_file, 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for i in range(len(out_list)):
	for j in range(len(out_list[i])):
		out_list[i][j] /= max_val
for i in range(len(out_list)):
	if out_list[i]:
		writer.writerow(out_list[i])
out_file.close()

