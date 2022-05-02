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
from neural_networks.create_custom_1 import data, test_loader, train_loader
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

PATH = "NN_1_5.pt"

if device == "cuda:0":
    net.load_state_dict(torch.load(PATH))
elif device == "cpu":
    net.load_state_dict(torch.load(PATH,map_location = torch.device('cpu')))

net.eval()
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
alpha = 0.1
ncal = 20
score = torch.tensor([])
qhat = np.zeros(n)


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
        
#%% Conformal prediction
alpha = 0.05
ncal = 34
#score = torch.tensor([])
alphas = np.arange(0,1,0.01)
n = len(alphas)
score = torch.tensor([])
qhat = np.zeros(n)

nN = 0
nP = 0

TN = np.zeros(100)
FP = np.zeros(100)
TP = np.zeros(100)
FN = np.zeros(100)

with torch.no_grad():
    for i,data in enumerate(test_loader):
        print("iter", i)

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
        if i < ncal:
            score = torch.cat((score,outputs[np.arange(8),labels]))
        else:
            if i == ncal:
                qhat = np.quantile(score,alphas,interpolation = "lower")
            elif i >= ncal:
                nN += len(np.where(labels==0)[0])
                nP += len(np.where(labels==1)[0])
                for j in range(100):
                    idx = np.where(outputs > qhat[j])
                    CP = [[]for i in range(8)]
                    for k in range(np.shape(idx)[1]):
                        CP[idx[0][k]].append(idx[1][k])
                    
                    for l,label in enumerate(labels):
                        if label == 0:
                            if 0 in CP[l]:
                                TN[j] += 1
                            if 1 in CP[l]:
                                FP[j] += 1
                        if label == 1:
                            if 1 in CP[l]:
                                TP[j] += 1
                            if 0 in CP[l]:
                                FN[j] += 1
                print("no defect")
                print("TN: ", TN[::10])
                print("FP: ", FP[::10])
                print("defect")
                print("TP: ", TP[::10])
                print("FN: ", FN[::10])
                #if len(np.where(labels==1)[0]) > 0:
                 #   break

TN /= nN
FP /= nN
TP /= nP
FN /= nP

#%% plots of conformal prediction
import tikzplotlib as tikz


thresh = 0.05

plt.plot(alphas, TN)
plt.plot(alphas, FP)
plt.plot(thresh*np.ones(100),np.arange(0,1,0.01))
plt.title("Target Class: No defect")
plt.xlabel(r'\alpha=0.05')
plt.ylabel("Frequency")
plt.legend(["TN","FP",r"\alpha"])
tikz.save("No_defect_CP.tex")
plt.show()


plt.plot(alphas, TP)
plt.plot(alphas, FN)
plt.plot(thresh*np.ones(100),np.arange(0,1,0.01))
plt.title("Target Class: Defect")
plt.xlabel("alpha")
plt.ylabel("Frequency")
plt.legend(["TP","FN",r"\alpha = 0.05"])
tikz.save("Defect_CP.tex")
plt.show()

#%%