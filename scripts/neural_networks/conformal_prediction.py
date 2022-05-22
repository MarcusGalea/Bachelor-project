# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:09:45 2022
Authors :
    Aleksander Svendstorp - s194323
    Marcus Galea Jacobsen - s194336

 Last edited : 21/05 -2022

 Script name : conformal_prediction . py
 Script function : Create figures for analysis on Conformal prediction sets
 
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



#%%

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%% Conformal prediction
alpha = 0.05

#size of calibration set
ncal = len(test_loader)//2


#define significance level space
alphas = torch.arange(0,1,0.01).to(device)
n = len(alphas)
score = torch.tensor([]).to(device)
qhat = np.zeros(n)

# number of negative, positive and total data
nN = 0
nP = 0
nD = 0


#performance metrics for each value of alpha
TN = np.zeros(100)
FP = np.zeros(100)
TP = np.zeros(100)
FN = np.zeros(100)

#amount of high quality prediction sets for each alpha
N_HC = np.zeros(100)

#sampling low quality data
low_conf = [[] for i in range(100)]

with torch.no_grad():
    for i,data in enumerate(test_loader):
        print("iter", i)

        #load data
        images, labels = data #range(0,250)
        #images -= avg_im #range(-250,250)
        images /= 255 #range(-1,1)
        #images += 1 #range(0,2)
        #images /= 2 #range(0,1)
        if device == "cuda:0": #send to GPU
            images = images.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)  
        outputs = net(images) #neural network estimates
        _, predictions = torch.max(outputs, 1)
        if i < ncal: #calibration data
            #score calculation in calibration data
            score = torch.cat((score,outputs[np.arange(8),labels]))
        else:
            if i == ncal: #calibration step. Calculate quantile for threshold
                qhat = torch.quantile(score,(ncal+1)*alphas/ncal,interpolation = "lower")
                
            elif i >= ncal: #test data
            
                nD += len(images)
                nN += len(torch.where(labels==0)[0])
                nP += len(torch.where(labels==1)[0])
                
                
                for j in range(100): #loop through significance levels
                    idx = torch.where(outputs > qhat[j]) #apply threshold on data
                    CP = [[]for i in range(8)]
                    for k in range(np.shape(idx[1])[0]):
                        #produce conformal prediction sets
                        CP[idx[0][k]].append(idx[1][k])
                    
                    for l,label in enumerate(labels):#loop through batch

                        if len(CP[l]) == 1: #high confidence data
                            N_HC[j] += 1
                            pred = CP[l][0]
                            
                            #performance mesurements of high confidence data
                            if pred == label and pred == 1:
                                TP[j] += 1
                            elif pred != label and pred == 1:
                                FP[j] += 1
                            elif pred == label and pred == 0:
                                TN[j] += 1
                            elif pred != label and pred == 0:
                                FN[j] += 1
                        else: #low confidence data 
                            low_conf[j].append(test_indices[i*8+l])
                            
                print("no defect")
                print("TN: ", TN[::10])
                print("FP: ", FP[::10])
                print("defect")
                print("TP: ", TP[::10])
                print("FN: ", FN[::10])
                #if len(np.where(labels==1)[0]) > 0:
                 #   break
#%% plots of conformal prediction
import tikzplotlib as tikz


alphas = alphas.cpu()
HCD = (TP+TN+FP+FN)/nD #proportion of HQ data

TNr = TN/(FP+TN)

TPr = TP/(TP+FN)

plt.plot(alphas, TPr)
plt.plot(alphas, TNr)
plt.plot(alphas, HCD)
#plt.plot(thresh*np.ones(100),np.arange(0,1,0.01))
plt.title("High confidence set")
plt.xlabel("alpha")
plt.ylabel("Frequency")
plt.legend(["TPR","TNR","High Conf Dat "])
tikz.save("CP_1_14.tex")
plt.show()
