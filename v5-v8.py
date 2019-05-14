import csv
basic_dir = '../'
data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
v5 = list(csv.reader(open(data_file, 'r')))
length = len(v5)
for i in range(length):
	v5[i] = list(map(float, v5[i]))
max_data = max(max(v5))
for i in range(length):
	for j in range(len(v5[i])):
		v5[i][j] = v5[i][j] / max_data
out_file = open(basic_dir + 'music-data-v8.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for i in range(length):
	writer.writerow(v5[i])
out_file.close()


