import csv
file = open('D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/labels-v5.csv', 'r')
reader = csv.reader(file)
label_list = list(reader)
length = len(label_list)
label_type = 0
history = []
one_hot = []
for item in label_list:
	same_as_one = False
	item = list(map(int, list(map(float, item))))
	for i in range(len(history)):
		if item == history[i]:
			one_hot.append(i)
			same_as_one = True
			break
	if not same_as_one:
		one_hot.append(label_type)
		label_type += 1
		history.append(item)
file.close()

out_file = open('D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/one-hot-label.csv', 'w', encoding='utf8', newline='')
writer = csv.writer(out_file)
writer.writerow(one_hot)

