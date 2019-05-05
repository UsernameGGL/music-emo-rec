import csv
import torch
file = open('D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/one-hot-label.csv', 'r')
labels = torch.Tensor(list(map(int, list(csv.reader(file))[0])))
idx = torch.max(labels)
print(idx)
