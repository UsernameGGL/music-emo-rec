import csv
file = open('D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/one-hot-label.csv', 'r')
labels = list(map(int, list(csv.reader(file))[0]))
print(len(labels))
