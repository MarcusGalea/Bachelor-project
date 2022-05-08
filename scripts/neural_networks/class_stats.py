# -*- coding: utf-8 -*-

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pathname = Path(dname)
parent_folder = pathname.parent.absolute()

PATH = str(parent_folder) + "\\NN_1_5.pt"

#get dataloader
from create_custom_1 import data, test_loader, train_loader, test_indices
from NN import net,device,Net,avg_im
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision.io import read_image

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


with torch.no_grad():
    for i,data in enumerate(test_loader):
        print(i)
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
        for label, prediction in zip(labels,predictions):
            
            # C
            if label == 1 and prediction == 1:
                C[0,0] += 1
            if label == 0 and prediction == 1:
                C[0,1] += 1
            if label == 1 and prediction == 0:
                C[1,0] += 1
            if label == 0 and prediction == 0:
                C[1,1] += 1
            print(C)