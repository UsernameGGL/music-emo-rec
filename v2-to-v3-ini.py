import csv
import numpy as np
import os
import statistics

pic_len = 256
interval = 3000

output_dir = 'E:/data/cal500/music-data-v3/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except(TypeError, ValueError):
        pass
    return False


# 读入音频数据，计算数据行数
input_file = open('../data/cal500/prodAudios_v2.txt', 'r')
input_slice = input_file.readlines()
create_inp_list = True
slice_num = 0
length = len(input_slice)
for i in range(length):
    line = input_slice[i].split(' ')
    if line[len(line) - 1] == '\n':
        line.pop()
    line = list(map(int, line))
    print('正在处理第' + str(i + 1) + '个文件')
    # print(i)
    # line = input_slice[i]
    line_len = len(line)
    print(line_len)
    sample_start = 0
    sample_len = pic_len * pic_len
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
        # for j in range(len(freq_data)):
        #     freq_data[j] = round(freq_data[j], 4)
        end_data = statistics.mean(freq_data[1:])
        sample_start += interval
        # #########################
        # notice!: time data - freq data - and then mean of freq[1:] in one row
        row_data = np.append(time_data, freq_data)
        row_data = np.append(row_data, end_data)
        writer.writerow(row_data)
    # writer.writerow(line[sample_start: line_len])
    out_file.close()
    out_file = open(output_dir + file_name, 'r')
    slice_reader = csv.reader(out_file)
    slices = list(slice_reader)
    cnt = 0
    for one_slice in slices:
        if not one_slice:
            continue
        cnt += 1
        cnt_2 = 0
        for num in one_slice:
            cnt_2 += 1
            if not is_number(num):
                print(cnt_2, 'not a number')
                print(num)
                print("processing the {} samples of {} slices".format(cnt, i+1))
    out_file.close()
