import csv
import os

basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
audio_path = basic_dir + 'music-data-v4.csv'
label_path = basic_dir + 'labels_v4_back.csv'
audio_file = open(audio_path, 'r')
label_file = open(label_path, 'r')
labels = list(csv.reader(label_file))
audio_dir = basic_dir + 'raw-data-v5/'
if not os.path.exists(audio_dir):
	os.makedirs(audio_dir)
error_idx = []
exp_num = 0
for i in range(3223):
	one_audio = audio_file.readline().split(',')
	audio_len = len(one_audio)
	if audio_len < 65536:
		error_idx.append(i)
insert_p = error_idx[0]
found_p = 1
len_error = len(error_idx)
audio_file.close()
label_file.close()
print('长度总共是{}'.format(insert_p))
print('发现了{}个坏点'.format(found_p))
print('len(error_idx)是{}'.format(len_error))
print('计算得到的长度是{}'.format(3223 - insert_p))
print('输出长度是', exp_num)
print(error_idx)
print(error_idx[0])
