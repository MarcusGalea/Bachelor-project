"""
Authors:
    Aleksander Svendstorp - s194323
    Marcus Galea Jacobsen - s194336
    
Last edited: 19/05-2022

Script name: NN.py
Script function: Script that defines our binary classifying CNN
"""

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
from torch import nn
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

class Net(nn.Module):
    def __init__(self,
                 kernw = 70, #width/height of convolution
                 kernlayers = 10, #number number of layers in first convolution (twice as many in second convolution)
                 l1 = 100, #number of outputs in first linear transformation
                 l2 = 50, #number of outputs in second linear transformation
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
        self.fc3 = nn.Linear(l2, 2)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
    
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
        x = F.relu(self.drop(self.fc1(x)))
        x = F.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x

net = Net(kernw = 70,
          kernlayers = 10,
          l1=100,
          l2=50,
          imagew = 300
          )

avg_im = read_image(direc + series+"_average_cell300.png")[0]

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

criterion = nn.CrossEntropyLoss(weight=w)

optimizer = torch.optim.Adam(net.parameters(),lr =0.0001)

