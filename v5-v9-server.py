import csv
basic_dir = '../'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
v5 = list(csv.reader(open(data_file, 'r')))
length = len(v5)
for i in range(length):
	v5[i] = list(map(float, v5[i]))
max_data = max(max(v5))
print('max_data is {}'.format(max_data))

out_list = [[] for i in range(3219)]
idx = 0
sum_num = 4096
need_num = 16
for i in range(length):
	out = []
	for j in range(need_num):
		out.append(sum(v5[i][j: j + sum_num]) / sum_num)
	out_list[idx] = out
	idx += 1
max_data = max(max(out_list))

out_file = open(basic_dir + 'music-data-v9.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for i in range(length):
	for j in range(need_num):
		out_list[i][j] = out_list[i][j] / max_data
	writer.writerow(out_list[i])	
out_file.close()


