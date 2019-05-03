import csv
from pydub import AudioSegment
import os
import numpy as np
import statistics

path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'CAL500_32kps/'
labelPath = path+'SegLabelHard/'

audioDir = os.listdir(audioPath)
labelDir = os.listdir(labelPath)

labelExport = []

i = 0
exportNum = 0
sample_num = 100
pic_len = 256
output_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/music-data-v4/'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
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
		labelExport.append(labels[j][2:20])
		start = float(labels[j][0])*1000
		end = float(labels[j][1])*1000
		oneSlice = audioFile[start: end]
		raw_data = [oneSlice.raw_data[k] for k in range(len(oneSlice.raw_data))]
		raw_data = np.array(raw_data)
		raw_data.shape = -1, 2
		for k in range(len(raw_data)):
			raw_data[k][0] = raw_data[k][0]*256 + raw_data[k][1]
		raw_data = raw_data[:, 0]
		sample_start = 0
		sample_len = pic_len * pic_len
		line_len = len(raw_data)
		interval = int((line_len - sample_len) / (sample_num - 1))
		file_name = '0'
		tmp_i = exportNum
		while tmp_i < 10000:
			tmp_i *= 10
			file_name += '0'
		file_name += str(exportNum + 1) + '.csv'
		out_file = open(output_dir + file_name, 'w', encoding='utf8', newline='')
		writer = csv.writer(out_file)
		while sample_start + sample_len <= line_len:
			time_data = raw_data[sample_start: sample_start + sample_len]
			freq_data = abs(np.fft.fft(time_data) / sample_len)
			for kkk in range(len(freq_data)):
				freq_data[kkk] = round(freq_data[kkk], 5)
			end_data = round(statistics.mean(freq_data[1:]), 5)
			sample_start += interval
	        # #########################
	        # notice!: time data - freq data - and then mean of freq[1:] in one row
			row_data = np.append(time_data, freq_data)
			row_data = np.append(row_data, end_data)
			writer.writerow(row_data)
		exportNum += 1
	i += 1
	labelFile.close()
	print('已经处理完了{}乘以2个文件诶'.format(i))

print('下面该写label_csv了')

laProFName = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/labels_v4.csv'
laExportFile = open(laProFName, 'w' , encoding='utf8', newline='')
laExportWriter = csv.writer(laExportFile)
for item in labelExport:
	if item != []:
		laExportWriter.writerow(item)
laExportFile.close()



