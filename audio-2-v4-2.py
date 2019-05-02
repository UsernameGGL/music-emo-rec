import csv
from pydub import AudioSegment
import os
import numpy as np

path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'CAL500_32kps/'
labelPath = path+'SegLabelHard/'

audioDir = os.listdir(audioPath)
labelDir = os.listdir(labelPath)

i = 0
exportNum = 0
sample_num = 100
pic_len = 256
output_path = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/music-data-v4.csv'
output_file = open(output_path, 'w', encoding='utf8', newline='')
output_writer = csv.writer(output_file)
for audioFName in audioDir:
	auPrefix = audioFName.split('.')[0]
	laPrefix = labelDir[i].split('.')[0]
	if auPrefix != laPrefix:
		print('error! files not in relevant.')
		os._exit(0)
	audioFile = AudioSegment.from_mp3(audioPath+audioFName)
	labelFile = open(labelPath+labelDir[i])
	labelReader = csv.reader(labelFile)
	labels = list(labelReader)
	for j in range(1, len(labels)):
		start = float(labels[j][0])*1000
		end = float(labels[j][1])*1000
		oneSlice = audioFile[start: end]
		# raw_data = [oneSlice.raw_data[k] for k in range(len(oneSlice.raw_data))]
		raw_data = list(oneSlice.raw_data)
		raw_data = np.array(raw_data)
		raw_data.shape = -1, 2
		for k in range(len(raw_data)):
			raw_data[k][0] = raw_data[k][0]*256 + raw_data[k][1]
		raw_data = raw_data[:, 0]
		output_writer.writerow(raw_data)
		exportNum += 1
	i += 1
	labelFile.close()
	print('已经处理完了{}乘以2个文件诶'.format(i))
output_file.close()


