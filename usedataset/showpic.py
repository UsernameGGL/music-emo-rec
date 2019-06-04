from MusicDataset import MusicDataFour
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
data_dir = basic_dir + 'raw-data-v5/'
# data_file = basic_dir + 'music-data-v5.csv'
label_file = basic_dir + 'labels-v5.csv'
train_set = MusicDataFour(data_dir, label_file, start=0, total=10)
train_loader = DataLoader(dataset=train_set,
                          batch_size=1, shuffle=True)

for i, data in enumerate(train_loader, 0):
	images, labels = data
	# images = Image(images)
	# plt.imshow(images)
	# images.show()
	print(data)
	break
