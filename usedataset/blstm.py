import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MusicDataset import Musicdata_LSTM
from MusicDataset import MusicDataThree
import sys
arg_len = len(sys.argv)
mode = sys.argv[1]
# Hyper-parameters
num_classes = 18
batch_size = 128
# basic_dir = 'D:/OneDrive-UCalgary/OneDrive - University of Calgary/data/cal500/'
basic_dir = '../'
label_file = basic_dir + 'labels-v5.csv'
if mode == 'shallow':
    data_file = basic_dir + 'music-data-v9.csv'
    train_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=2560, mode='shallow')
    test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=2560, total=3219, mode='shallow')
    prefix = 'shallow-lstm'
    sequence_length = 4
    input_size = 4
    hidden_size = 32
    num_layers = 2
    num_epochs = 50
    learning_rate = 0.3
else:
    data_file = basic_dir + 'music-data-v5.csv'
    train_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=2560)
    test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=2560, total=3219)
    prefix = 'deep-lstm'
    sequence_length = 256
    input_size = 256
    hidden_size = 64
    num_layers = 60
    num_epochs = 2000
    learning_rate = 30
record_file = 'record-' + prefix + '.txt'
model_path = prefix + '.pt'
record_num = 10
# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')



# MNIST dataset

train_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=0, total=2560)
test_set = Musicdata_LSTM(data_file=data_file, label_file=label_file, start=2560, total=3219)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100,
        threshold=1e-6, factor=0.5, min_lr=1e-6)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device).view(-1, num_classes)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        scheduler.step(loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % record_num == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss / record_num))
            with open(record_file, 'a') as f:
                f.write(str(running_loss / record_num) + '\n')
            running_loss = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), model_path)

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    sigmoid = nn.Sigmoid()
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device).view(-1, num_classes)
        outputs = model(images)
        outputs = sigmoid(outputs)
        one_correct = 0
        for i in np.arange(0.3, 0.7, 0.01):
            tmp_outputs = outputs.clone()
            tmp_outputs[tmp_outputs > i] = 1
            tmp_outputs[tmp_outputs <= i] = 0
            tmp_correct = (tmp_outputs.data == labels).sum().item()
            if tmp_correct > one_correct:
                one_correct = tmp_correct
                # threshold = i
                r_outputs = tmp_outputs.clone()
        # print('--------threshold', threshold)
        outputs = r_outputs
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) * num_classes
        correct += (outputs == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 
    with open(record_file, 'a') as f:
        f.write(str(100 * correct / total) + '\n')
