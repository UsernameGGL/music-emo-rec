import torch
batch_size = 100
label_len = 18
correct_v4 = 0
total_v4 = 0
for k in range(batch_size):
	_, index = torch.sort(outputs[k])
	emotion_num = torch.sum(labels[k])
	total_v4 += emotion_num
	for kk in range(emotion_num):
		if labels[k][index[kk]] == 1:
			correct_v4 += 1
print('My_Accuracy_4 of the network on the test images: %d %%' % (
             100 * correct_v4 / total_v4))




