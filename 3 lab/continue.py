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

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE*IMAGE_SIZE, 4*IMAGE_SIZE*IMAGE_SIZE)
        self.fc2 = nn.Linear(4*IMAGE_SIZE*IMAGE_SIZE, 4*NUM_CLASSES)
        self.fc3 = nn.Linear(4*NUM_CLASSES, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE*IMAGE_SIZE)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
    
def GetItem(y):

    img_name = ""
    for i in range(len(files)):
        if files[i][1] == y:
            idx = i
            break

    img_name = files[idx][0]
    img = cv2.imread(img_name, cv2.COLOR_RGB2GRAY)
    image = img
    image = cv2.resize(image,(IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.bitwise_not(image) / 255
    image = np.asarray(image).astype(float)#.reshape(3,self.img_size,self.img_size)

    target = [0 for i in range(NUM_CLASSES)]
    target[files[idx][1]] = 1
    target = torch.FloatTensor(target)
    
    image = torch.FloatTensor(image[:,:,0])

    return image,target,files[idx][1]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
path_to_folder = '.\data' # путь к папке с данными
name_folders = [x[0].split('\\')[-1] for x in os.walk(path_to_folder)]
name_folders = name_folders[1:]

NUM_CLASSES = len(name_folders)
MAX_FILES = 3
IMAGE_SIZE = 32

class_idx = [i for i in range(len(name_folders))]
dict_folders = {name_folders[i]:class_idx[i] for i in range(len(class_idx))}
print(dict_folders)

file_names = []
class_labels = []
for path, subdirs, files in os.walk(path_to_folder):
    for (idx,name) in enumerate(files):
        if(idx < MAX_FILES):
            file_names.append(os.path.join(path, name))
            class_labels.append(dict_folders[path.split('\\')[-1]])

files = [[file_names[i],class_labels[i]] for i in range(len(file_names))]

my_model = Net().to(device)
my_model.load_state_dict(torch.load('./models/mlp_model_22_classes_1000.pth'))
my_model.eval()

value = "1 0 + 3 - 6" # Строка для создания массива картинок
end = ""
for i in value.split(" "):
    (X_train, y_train,class_idx) = GetItem(dict_folders[i]) # Получаем текущую картинку 
    result = my_model(X_train) # Предсказываем поведение
    temp = list(dict_folders.keys())[list(dict_folders.values()).index(int(torch.argmax(result)))] # Вычлисляем действительное значение данной модели
    end += temp # Сохраняем
    print(temp, end="") # Выводим

print(f"={eval(end)}") # Расчет значения (не будет работать для * и букв)