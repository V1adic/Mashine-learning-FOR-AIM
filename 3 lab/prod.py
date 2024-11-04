import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt

import os

import pandas as pd
from skimage import io,morphology
from PIL import Image, ImageOps
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision import transforms
from random import shuffle
import cv2 

class NumericDataset(Dataset):

    def __init__(self, root_dir, img_size, num_classes, transform = None):
        self.root_dir = root_dir
        self.img_size = img_size # !!! внимательно при изменении размеров изображения
        self.transform = transform
        self.num_classes = num_classes
        file_names = []
        class_labels = []
        for path, subdirs, files in os.walk(path_to_folder):
            for (idx,name) in enumerate(files):
                if(idx < MAX_FILES):
                    file_names.append(os.path.join(path, name))
                    class_labels.append(dict_folders[path.split('\\')[-1]])
        self.files = [[file_names[i],class_labels[i]] for i in range(len(file_names))]#!!!
        shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx][0]

        img = cv2.imread(img_name, cv2.COLOR_RGB2GRAY)
        image = img
        image = cv2.resize(image,(self.img_size, self.img_size))
        image = cv2.bitwise_not(image) / 255
        image = np.asarray(image).astype(float)#.reshape(3,self.img_size,self.img_size)

        target = [0 for i in range(self.num_classes)]
        target[self.files[idx][1]] = 1
        target = torch.FloatTensor(target)
        
        image = torch.FloatTensor(image[:,:,0])
        if self.transform:
            image = self.transform(image)

        return image,target,self.files[idx][1]
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE*IMAGE_SIZE, 4*IMAGE_SIZE*IMAGE_SIZE)
        #self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4*IMAGE_SIZE*IMAGE_SIZE, 4*NUM_CLASSES)
        #self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(4*NUM_CLASSES, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE*IMAGE_SIZE)
        #x = F.sigmoid(self.fc1(x))
        x = self.fc1(x)
        #x = self.fc1_drop(x)
        #x = F.sigmoid(self.fc2(x))
        x = self.fc2(x)
        #x = self.fc2_drop(x)
        #return F.softmax(self.fc3(x), dim=1)
        return self.fc3(x)
    
def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    epoch_loss = 0
    k = 0    
    # Loop over each batch from the training set
    for batch_idx, (data, target,idx_class) in enumerate(num_train_dataloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        #print(output,target)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        k+=1
        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step() 
      
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Epoch_Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(num_train_dataloader.dataset),
                100. * batch_idx / len(num_train_dataloader), loss.data.item(), epoch_loss))
    return epoch_loss / k


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
path_to_folder = '.\data' # путь к папке с данными
name_folders = [x[0].split('\\')[-1] for x in os.walk(path_to_folder)]
name_folders = name_folders[1:]
print(name_folders)
NUM_CLASSES = len(name_folders)
MAX_FILES = 3

class_idx = [i for i in range(len(name_folders))]
dict_folders = {name_folders[i]:class_idx[i] for i in range(len(class_idx))}

file_names = []
class_labels = []
for path, subdirs, files in os.walk(path_to_folder):
    for name in files:
        file_names.append(os.path.join(path, name))
        class_labels.append(dict_folders[path.split('\\')[-1]])

batch_size = 10
IMAGE_SIZE = 32
num_train_dataloader = DataLoader(NumericDataset(path_to_folder,IMAGE_SIZE,NUM_CLASSES), batch_size=batch_size, shuffle=True)

for (X_train, y_train,class_idx) in num_train_dataloader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

epoch_start = 0
epochs = 1000
path_model_save = './models/'

model = Net().to(device) #!!!

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

model.train()
lossv = []
for epoch in range(epoch_start, epochs + 1):
    lossv.append(train(epoch))
    
torch.save(model.state_dict(), path_model_save+'mlp_model_22_classes_'+str(epoch)+'.pth')
    #validate(lossv, accv)
    #train(epoch, model_CNN)
    #validate(lossv_CNN, accv_CNN)