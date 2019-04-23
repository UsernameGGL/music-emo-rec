import csv

file = open('C:/Users/guoliang.gao1/OneDrive - University of Calgary/data/cal500/music-data/000001.csv')
slices = list(csv.reader(file))
for line in slices:
    print(len(line))
