from torch.utils.data import Dataset
import csv
import os
import torch
class MusicDataOne(Dataset):
	def __init__(self, data_dir, label_file, transform=None, pic_len=256, start=0, total=3223):
		self.start = start
		self.total = total

		label_file = open(label_file, 'r')
		label_reader = csv.reader(label_file)
		reader_list = list(label_reader)
		create_labels = True
		for line in reader_list:
		    if line == []:
		        continue
		    if create_labels:
		        self.labels = [list(map(int, line))]
		        create_labels = False
		    else:
		        self.labels.append(list(map(int, line)))
		self.label_len = len(self.labels[0])
		label_file.close()

		self.file_names = os.listdir(path=data_dir)
		self.sample_len = pic_len*pic_len
		self.data_dir = data_dir
		self.pic_len = pic_len

	def __len__(self):
		return (self.total - self.start) * 100

	def __getitem__(self, idx):
		file = open(self.file_names[int(idx/100)])
		rows = list(csv.reader(file))
		row = list(map(float, rows[idx % 100]))
		time_data = row[0: self.sample_len]
		freq_data = row[self.sample_len: self.sample_len * 2]
		freq_data[0] = row[self.sample_len * 2]
		time_data = torch.Tensor(time_data).view(1, self.pic_len, self.pic_len)
		freq_data = torch.Tensor(freq_data).view(1, self.pic_len, self.pic_len)
		data = torch.cat((time_data, freq_data))
		label = self.labels[int(idx/100)]
		if self.transform:
			data = self.transform(data)
		return data, label

class MusicDataTwo(object):
	"""docstring for MusicDataTwo"""
	def __init__(self, data_dir, label_file, transform=None, pic_len=256):
		super(MusicDataTwo, self).__init__()

		label_file = open(label_file, 'r')
		label_reader = csv.reader(label_file)
		reader_list = list(label_reader)
		create_labels = True
		for line in reader_list:
		    if line == []:
		        continue
		    if create_labels:
		        self.labels = [list(map(int, line))]
		        create_labels = False
		    else:
		        self.labels.append(list(map(int, line)))
		self.label_len = len(self.labels[0])
		self.len = len(self.labels)
		label_file.close()

		self.file_names = os.listdir(path=data_dir)
		self.sample_len = pic_len*pic_len
		self.data_dir = data_dir
		self.pic_len = pic_len

	def __len__(self):
		return self.len * 100

	def __getitem__(self, idx):
		file = open(self.file_names[int(idx/100)])
		rows = list(csv.reader(file))
		row = list(map(float, rows[idx % 100]))
		time_data = row[0: self.sample_len]
		freq_data = row[self.sample_len: self.sample_len * 2]
		freq_data[0] = row[self.sample_len * 2]
		time_data = torch.Tensor(time_data).view(1, self.pic_len, self.pic_len)
		freq_data = torch.Tensor(freq_data).view(1, self.pic_len, self.pic_len)
		data = torch.cat((time_data, freq_data))
		label = self.labels[int(idx/100)]
		if self.transform:
			data = self.transform(data)
		return data, label
		
