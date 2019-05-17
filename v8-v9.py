import csv
basic_dir = '../'
data_file = basic_dir + 'music-data-v8.csv'
v8 = list(csv.reader(open(data_file, 'r')))
length = len(v8)
for i in range(length):
	v8[i] = list(map(float, v8[i]))
max_data = max(max(v8))

sum_num = 4096
out_file = open(basic_dir + 'music-data-v9.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for row in v8:
	length = len(row)
	i = 0
	out = []
	while i * sum_num + sum_num < length:
		out.append(sum(row[i * sum_num: i * sum_num + sum_num]) / sum_num)
		i += 1
	writer.writerow(out)
out_file.close()

