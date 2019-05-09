import csv
import os
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_file = basic_dir + 'music-data-v6.csv'
# label_file = basic_dir + 'labels-v5.csv'
# data_file = basic_dir + 'music-data-v4.csv'
data_file = basic_dir + 'music-data-v6.csv'
v6 = list(csv.reader(open(data_file, 'r')))
length = len(v6)
for i in range(length):
	v6[i] = list(map(float, v6[i]))
max_data = max(max(v6))
for i in range(length):
	for j in range(len(v6[i])):
		v6[i][j] = v6[i][j] / max_data
out_file = open(basic_dir + 'music-data-v7.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
for i in range(length):
	writer.writerow(v6[i])
out_file.close()

