# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:09:45 2022

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
from data_sets.create_custom_1 import data, test_loader, train_loader
from neural_networks.NN import *
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
PATH = "NN_1_6.pt"

if device == "cuda:0":
    net.load_state_dict(torch.load(PATH))
elif device == "cpu":
    net.load_state_dict(torch.load(PATH,map_location = torch.device('cpu')))

#net.eval()
#%% Testing accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        images -= avg_im #range(-250,250)
        images /= 255 #range(-1,1)
        images += 1 #range(0,2)
        images /= 2 #range(0,1)
        if device == "cuda:0":
            images = images.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)
        # calculate outputs by running images through the network
        outputs = net(images)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(100*correct//total)

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

#%% Testing class accuracy
classes = ('No Defect','Defect')
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
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
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

        # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
#%% Confusion matrix
#TP FP
#FN TN
C = np.array([[0,0],[0,0]])
k = 0
with torch.no_grad():
    for data in test_loader:
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
            if label == 1 and prediction == 1:
                C[0,0] += 1
            if label == 0 and prediction == 1:
                C[0,1] += 1
            if label == 1 and prediction == 0:
                C[1,0] += 1
            if label == 0 and prediction == 0:
                C[1,1] += 1
        k += 1
        print(k)

print(C)