# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:03:24 2022

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

class Net(nn.Module):
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
        self.fc3 = nn.Linear(l2, 1)
        self.sig = nn.Sigmoid()
        self.init_weights(weights,biases)
    
    def init_weights(self,weights,biases):
        with torch.no_grad():
            if len(weights) > 0:
                self.conv1.weight = nn.Parameter(weights)
                print(weights)
            if len(biases) > 0:
                self.conv1.bias = nn.Parameter(biases)
                
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
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

net = Net(kernw = 50,
          l1=80,
          l2=40,
          weights = weights, 
          biases = biases
          )

avg_im = read_image(direc + series+"_average_cell.png")[0]

device = "cpu"
"""
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        print(device)
        net = nn.DataParallel(net)
"""
#net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr =0.00709694)

#%% show kernels
#plt.imshow(net.conv1.weight.detach().numpy()[0][0],cmap = "gray")
#plt.imshow(convolutions[8][0],cmap = "gray")
plt.imshow(avg_im,cmap = "gray")
#%% Training

import multiprocessing as mp
import torch.multiprocessing as mp_t

def main():
    printfreq = 20
    N = len(train_loader)
    for epoch in range(2):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, datas in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = datas
            inputs -= avg_im
            inputs = inputs.type(torch.cuda.FloatTensor)#.to(device)
            #labels = labels.to(device)
    
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



main()

#%% Testing
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(100*correct//total)

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

#%%
classes = ('No Defect','Defect')
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
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
    
#%% save/load model 
torch.save(net.state_dict(), dname)

net = Net()
net.load_state_dict(torch.load(dname))
net.eval()

#%% plots
plt.imshow(net.conv1.weight[5][0].cpu().detach().numpy(),cmap = "gray")