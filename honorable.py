import csv
input_file = open('./data/cal500/prodAudios_v2.txt', 'r')
input_slice = input_file.readlines()
cnt = 0
for line in input_slice:
	cnt += 1
	line = line.split(' ')
	line.pop()
	line = list(map(int, line))
	# print(line[10])
	ll = len(line)
	if ll < 65636:
		print(ll)
		print('---第{}个文件'.format(cnt))
