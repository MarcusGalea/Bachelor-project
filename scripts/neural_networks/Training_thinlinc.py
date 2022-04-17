# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:15:49 2022

@author: aleks
"""

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.io import read_image
from math import ceil, floor
import pandas as pd
from pytorch_metric_learning.samplers import MPerClassSampler
from math import floor
import random
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from functools import partial
import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
#os.chdir(dname)


pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)


# %%
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

global datadir
global labels
global images
global avg_im

user = "HPC"
if user == "Aleksander":
    direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\"
    series = r"AllSeries\\"
    images = r"CelssCorr_resize\\"
    labels = r"all_labels.csv"
    
    
elif user == "Marcus":
    direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
    series = r"AllSeries\\"
    images = "CellsCorr_resize"
    labels = "labels.csv"
    
elif user == "HPC":
    direc = '/zhome/35/5/147366/Desktop/'
    series = ''
    images = 'CellsCorr_resize'
    labels = 'labels.csv'


datadir = direc+series
avg_im = read_image(datadir +"_average_cell.png")[0]

#from data_sets.create_custom_1 import CustomImageDataset    
#%%
class Net(nn.Module):
    def __init__(self,
                 kernw=30,  # width/height of convolution
                 # number number of layers in first convolution (twice as many in second convolution)
                 kernlayers=6,
                 l1=120,  # number of outputs in first linear transformation
                 l2=84,  # number of outputs in second linear transformation
                 imagew=400,  # width/height of input image
                 drop_p = 0.5 # dropout rate
                 ):
        super().__init__()
        # third arg: remove n-1 from img dim
        self.conv1 = nn.Conv2d(1, kernlayers, kernw)
        self.pool = nn.MaxPool2d(2, 2)  # divide img dim with n
        self.conv2 = nn.Conv2d(kernlayers, 2*kernlayers, kernw//2)
        self.fc1 = nn.Linear(
            (((imagew-kernw)//2-kernw//2)//2)**2*2*kernlayers, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 2)
        self.drop = nn.Dropout(drop_p)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net(kernw=90, kernlayers=10, l1=100, l2=50, imagew=400, drop_p=0.5)

if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        print(device)
        net = nn.DataParallel(net)

net.to(device)

w = torch.tensor([1.,40.])
if device == "cuda:0":
    w = w.type(torch.cuda.FloatTensor)#.to(device)

criterion = nn.CrossEntropyLoss(weight=w)

optimizer = torch.optim.Adam(net.parameters(),lr =0.0001)
    
#%%
sample_test = False

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image[0:1].type(torch.FloatTensor), label



#def load_data(data_dir= None):
data = CustomImageDataset(annotations_file = direc+series+labels,
                          img_dir = direc+series+images,
                          transform = transforms.RandomVerticalFlip())


labels = data.img_labels.to_numpy()[:,1]



#seed #DON'T CHANGE, OR ALL PRINCIPAL COMPONENTS MUST BE REMADE
#random.seed(10)

#initialize sizes
N = len(data)
train_size = 0.8 #proportion of training data
batch_size = 8 #batch size
m = batch_size//2 #amount of data per class in every batch


#randomly shuffle indices
indices = list(range(N))
random.shuffle(indices)
#create indices for both training data and test data
train_indices = indices[:floor(train_size*N)]
test_indices = indices[floor(train_size*N):]


#split data into training data and test data
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)
train_labels = labels[train_indices]
test_labels = labels[test_indices]

#create sampler for each set of data, s.t each batch contains m of each class
train_sampler = MPerClassSampler(train_labels, m, batch_size=batch_size, length_before_new_iter=100000)
test_sampler = MPerClassSampler(test_labels, m, batch_size=batch_size, length_before_new_iter=100000)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}



#dataloader for training data
train_loader = DataLoader(
    train_split,
    shuffle=False,
    sampler = train_sampler,
    batch_size=batch_size,
    **kwargs
)

#dataloader for test data
test_loader = DataLoader(
    test_split,
    shuffle=False,
    #sampler = test_sampler,
    batch_size=batch_size,
    **kwargs
)

#%%
printfreq = 20
N = len(train_loader)


table = []
for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        #remove average image to remove lines
        inputs, labels = datas #range = (0,250)
        inputs -= avg_im #range = (-250,250)
        inputs /= 255 #range = (-1,1)
        inputs += 1 # range = (0,2)
        inputs /= 2 # range = (0,1)
        if device == "cuda:0":
            inputs = inputs.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)
        #labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % printfreq == printfreq-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            table.append([epoch +1, i+1, running_loss / printfreq])
            running_loss = 0.0

print('Finished Training')
PATH = "NN_1_12.pt"
torch.save(net.state_dict(), PATH)
pd.DataFrame(table).to_csv(r"zhome\35\5\147366\Desktop\loss_1_12.csv",header = None, index = None)