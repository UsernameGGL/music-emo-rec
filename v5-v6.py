import csv
import os
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_dir = basic_dir + 'raw-data-v5/'
# label_file = basic_dir + 'labels-v5.csv'
# data_file = basic_dir + 'music-data-v4.csv'
file_names = os.listdir(data_dir)
data_file = basic_dir + 'music-data-v6.csv'
out_file = open(data_file, 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
max_data = 0
cnt = 0
idx = 0
for data_file in file_names:
	row = list(csv.reader(open(data_dir + data_file, 'r')))[0]
	row = list(map(int, row))
	length = len(row)
	# i = 0
	# out = []
	# while i * 4096 + 4096 < length:
	# 	out.append(sum(row[i * 4096: i * 4096 + 4096]) / 4096)
	# 	i += 1
	# writer.writerow(out)
	cnt += 1
	tmp = max(row)
	if tmp > max_data:
		max_data = tmp
		idx = cnt
print(max_data)
print(idx)
out_file.close()

