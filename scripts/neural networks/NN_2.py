# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:06:46 2022

@author: Marcu
"""
import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)

#get dataloader
from data_sets.create_custom_1 import train_loader,test_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision.io import read_image


#%%

direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"

series = r"AllSeries\\"
kernel = r"Kernels\PC\\"
defect = r"All Defects\\"

class Net_2(nn.Module):
    def __init__(self,
                 kernw = 50, #width/height of convolution
                 kernlayers = 10, #number number of layers in first convolution (twice as many in second convolution)
                 l1 = 120, #number of outputs in first linear transformation
                 l2 = 84, #number of outputs in second linear transformation
                 weights = None,
                 biases = None,
                 imagew = 300, #width/height of input image
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, kernlayers, kernw) #third arg: remove n-1 from img dim
        self.sig = nn.Sigmoid()
        self.init_weights(weights,biases)
    
    def init_weights(self,weights,biases):
        with torch.no_grad():
            if len(weights) > 0:
                self.conv1.weight = nn.Parameter(weights)
            if len(biases) > 0:
                self.conv1.bias = nn.Parameter(biases)
                
    def forward(self, x):
        x = self.conv1(x)
        return x



PC_link = direc+series+kernel+defect

k = -1
weights = torch.zeros(10,1,50,50)
biases = torch.zeros(10)

for PC in os.listdir(PC_link):
    PC = read_image(PC_link + PC)
    PC = PC[0:1].type(torch.FloatTensor)
    PC /= np.linalg.norm(PC)
    if k == -1:
        avg = PC.reshape(-1,1).squeeze()
        k+= 1
        continue
    weights[k] = PC
    biases[k] = torch.dot(-PC.reshape(-1,1).squeeze(),avg)
    k+= 1

net = Net_2(kernw = 50,
          l1=80,
          l2=40,
          weights = weights, 
          biases = biases
          )

avg_im = read_image(direc + series+"_average_cell.png")[0]

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr =0.00709694)

for i, datas in enumerate(train_loader):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = datas
    inputs -= avg_im
    break
#%% show kernels

#first input image
plt.imshow(inputs[0][0],cmap = "gray")
#%% first convolutional channel
plt.imshow(net.conv1.weight.detach().numpy()[4][0],cmap = "gray")
#%% convolution between input and PC
plt.imshow(net(inputs)[1][0].detach().numpy(),cmap = "gray")


#%% Training

printfreq = 20
N = len(train_loader)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = datas
        inputs -= avg_im

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.reshape(1,-1),labels.type(torch.float).reshape(1,-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % printfreq == printfreq-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            running_loss = 0.0

print('Finished Training')
