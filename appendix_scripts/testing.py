"""
Authors:
    Aleksander Svendstorp - s194323
    Marcus Galea Jacobsen - s194336
    
Last edited: 19/05-2022

Script name: testing.py
Script function: Script that test a binary classifying CNN and creates class statistics for a CNNs performance
"""

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pathname = Path(dname)
parent_folder = pathname.parent.absolute()


PATH = str(parent_folder) + "\\NN_1_14.pt"

#get dataloader
from create_custom_1 import data, test_loader, train_loader,direc,series,test_split,test_indices, test_classes,test_titles,labeldf
from NN import device,Net,avg_im
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pandas as pd


net = Net(kernw = 70,
          kernlayers = 10,
          l1=100,
          l2=50,
          imagew = 300
          )

device = "cpu"

if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        print(device)
        net = nn.DataParallel(net)

net.to(device)
if device == "cuda:0":
    net.load_state_dict(torch.load(PATH))
elif device == "cpu":
    net.load_state_dict(torch.load(PATH,map_location = torch.device('cpu')))
net.eval()

#%% Confusion matrix
#TP FP
#FN TN
C = np.array([[0,0],[0,0]])
k = 0
alpha = 0.1
ncal = 20
score = torch.tensor([])
classes = pd.read_csv(direc+series+"labels_MC.csv")


Nc = [0,0,0,0]
predc = [0,0,0,0]

with torch.no_grad():
    for i,data in enumerate(test_loader):
        images, labels = data #range(0,250)
        images -= avg_im #range(-250,250)
        images /= 255 #range(-1,1)
        images += 1 #range(0,2)
        images /= 2 #range(0,1)
        if device == "cuda:0":
            images = images.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)  
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for j,info in enumerate(zip(labels,predictions)):
            label, prediction = info
            # C
            if label == 1 and prediction == 1:
                C[0,0] += 1
            if label == 0 and prediction == 1:
                C[0,1] += 1
            if label == 1 and prediction == 0:
                C[1,0] += 1
            if label == 0 and prediction == 0:
                C[1,1] += 1
            #statistics
            MC = test_classes[i*8+j]
            MC = MC.split('[')[1].split(']')[0].split(',')

            for k,c in enumerate(MC):
                c = float(c)
                if c == 1.0:
                    Nc[k] += 1
                    if int(prediction) == 1:
                        predc[k] += 1
        
#%% statistics
import tikzplotlib

predc = np.array(predc)
Nc = np.array(Nc)

plt.bar(["Crack A"," Crack B","Crack C","Finger Failure"],predc/Nc)
plt.ylabel("Frequency [%]")
plt.title("Relative amount of errors caught by neural network")
tikzplotlib.save("Statistics_1_14.tex")
plt.show()