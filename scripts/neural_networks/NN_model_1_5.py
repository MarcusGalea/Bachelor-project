# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:08:24 2022

@author: aleks
"""

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)

#get dataloader
from data_sets.create_custom_1_5 import train_loader,test_loader
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
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\"

series = r"AllSeries\\"
kernel = r"Kernels\PC\\" #Opdater
defect = r"All Defects\\" #Opdater

class Net_model_1_5(nn.Module):
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
        self.pool = nn.MaxPool2d(2, 2) #divide img dim with n
        self.conv2 = nn.Conv2d(kernlayers, 2*kernlayers, kernw//2)
        self.fc1 = nn.Linear((((imagew-kernw)//2-kernw//2)//2)**2*2*kernlayers, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2,4)
        """
        self.fcA = nn.Linear(l2, 2)
        self.fcB = nn.Linear(l2, 2)
        self.fcC = nn.Linear(l2, 2)
        self.fcF = nn.Linear(l2, 2)
        """
        self.sig = nn.Sigmoid()
        #self.init_weights(weights,biases)
    
    def init_weights(self,weights,biases):
        with torch.no_grad():
            if len(weights) > 0:
                self.conv1.weight = nn.Parameter(weights)
            if len(biases) > 0:
                self.conv1.bias = nn.Parameter(biases)
                
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        """
        A = self.fcA(x)
        B = self.fcB(x)
        C = self.fcC(x)
        Fin = self.fcF(x)
        return A,B,C,Fin
        """
        x = self.fc3(x)
        return x

"""
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
"""
net = Net_model_1_5(kernw = 50,
          l1=80,
          l2=40,
          )

avg_im = read_image(direc + series+"_average_cell.png")[0]

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        print(device)
        net = nn.DataParallel(net)

net.to(device)


w = torch.tensor([1.,10.])
if device == "cuda:0":
    w = w.type(torch.cuda.FloatTensor)#.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(),lr =0.0001)

#%% Training
printfreq = 20
N = len(train_loader)

table = []
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = datas #range(0,250)
        """
        labelA = labels[:,0]
        labelB = labels[:,1]
        labelC = labels[:,2]
        labelF = labels[:,3]
        """
        inputs -= avg_im #range(-250,250)
        inputs /= 255 #range(-1,1)
        inputs += 1 #range(0,2)
        inputs /= 2 #range(0,1)
        if device == "cuda:0":
            inputs = inputs.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)
        #labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        """
        outA = outputs[0]
        outB = outputs[1]
        outC = outputs[2]
        outF = outputs[3]
        
        lossA = criterion(outA,labelA)
        lossB = criterion(outB,labelB)
        lossC = criterion(outC,labelC)
        lossF = criterion(outF,labelF)
        """
        
        loss = criterion(outputs,labels)
    
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        """
        running_lossA += lossA.item()
        running_lossB += lossB.item()
        running_lossC += lossC.item()
        running_lossF += lossF.item()
        """

        if i % printfreq == printfreq-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            table.append([epoch +1, i+1, running_loss / printfreq])
            running_loss = 0.0

print('Finished Training')