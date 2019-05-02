import csv
import numpy as np
import os
import statistics

pic_len = 256
sample_num = 100

output_dir = 'E:/data/cal500/music-data-v4-back/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 读入音频数据，计算数据行数
input_file = open('E:/data/cal500/music-data-v4.csv', 'r')
reader = csv.reader(input_file)
input_slice = list(reader)
create_inp_list = True
slice_num = 0
length = len(input_slice)
for i in range(length):
    line = list(map(int, input_slice[i]))
    print('正在处理第' + str(i + 1) + '个文件')
    line_len = len(line)
    print(line_len)
    sample_start = 0
    sample_len = pic_len * pic_len
    interval = int((line_len - sample_len) / sample_num)
    tmp_i = i + 1
    file_name = '0'
    while tmp_i < 10000:
        tmp_i *= 10
        file_name += '0'
    file_name += str(i + 1) + '.csv'
    out_file = open(output_dir + file_name, 'w', encoding='utf8', newline='')
    writer = csv.writer(out_file)
    while sample_start + sample_len <= line_len:
        time_data = line[sample_start: sample_start + sample_len]
        freq_data = abs(np.fft.fft(time_data) / sample_len)
        for kkk in range(len(freq_data)):
            freq_data[kkk] = round(freq_data[kkk], 5)
        end_data = round(statistics.mean(freq_data[1:]), 5)
        sample_start += interval
        row_data = np.append(time_data, freq_data)
        row_data = np.append(row_data, end_data)
        writer.writerow(row_data)
    # writer.writerow(line[sample_start: line_len])
    out_file.close()
