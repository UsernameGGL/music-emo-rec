import os
import csv

pic_len = 256
path = 'E:/data/cal500/music-data/'
sliceDir = os.listdir(path=path)
sample_len = pic_len * pic_len
slice_num = 0


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


for sliceFNama in sliceDir:
    slice_file = open(path + sliceFNama, 'r')
    slice_reader = csv.reader(slice_file)
    slices = list(slice_reader)
    slice_num += 1
    cnt = 0
    for one_slice in slices:
        if not one_slice:
            continue
        cnt += 1
        # one_slice = list(map(float, one_slice))
        cnt_2 = 0
        for num in one_slice:
            cnt_2 += 1
            if not is_number(num):
                print(cnt_2)
                print(num)
                print("processing the {} samples of {} slices".format(cnt, slice_num))

# path = 'E:/data/cal500/music-data/'
# sliceDir = os.listdir(path=path)
# pic_len = 256
# sample_len = pic_len*pic_len
# slice_file = open(path + '000564.csv')
# slice_reader = csv.reader(slice_file)
# slices = list(slice_reader)
# cnt = 0
# for one_slice in slices:
#     if not one_slice:
#         continue
#     cnt += 1
#     if cnt == 17:
#         print(len(one_slice))
#         cnt_2 = 0
#         for num in one_slice:
#             cnt_2 += 1
#             if not is_number(num):
#                 print(cnt_2)
#                 print(num)
#
#         break
