basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
audio_path = basic_dir + 'music-data-v4.csv'
audio_file = open(audio_path, 'r')
lines = audio_file.readlines()
audio_file.close()
audio_file = open(audio_path, 'r')
line = audio_file.readline()
print(line == lines[0])
line = audio_file.readline()
print(line == lines[0])
