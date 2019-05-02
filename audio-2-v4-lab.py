import csv
from pydub import AudioSegment
import os

path = 'F:/a-workspace/python/datasets/CAL500_32kps/'
audioPath = path+'CAL500_32kps/'
labelPath = path+'SegLabelHard/'

audioDir = os.listdir(audioPath)
labelDir = os.listdir(labelPath)

labelExport = []

i = 0
exportNum = 0
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
		exportNum += 1
	i += 1
	labelFile.close()
	print('已经处理完了{}乘以2个文件诶'.format(i))


print('下面该写label_csv了')

laProFName = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/labels_v4_back.csv'
laExportFile = open(laProFName, 'w' , encoding='utf8', newline='')
laExportWriter = csv.writer(laExportFile)
for item in labelExport:
	if item != []:
		laExportWriter.writerow(item)
laExportFile.close()
