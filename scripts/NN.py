# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:03:24 2022

@author: Marcu
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from data_sets.create_custom_1 import train_loader,test_loader

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib as plt

#%%
class Net(nn.Module):
    def __init__(self,kernw = 30,kernlayers = 6,imagew = 300):
        super().__init__()
        self.conv1 = nn.Conv2d(1, kernlayers, kernw) #third arg: remove n-1 from img dim
        self.pool = nn.MaxPool2d(2, 2) #divide img dim with n
        self.conv2 = nn.Conv2d(6, 2*kernlayers, kernw//2)
        self.fc1 = nn.Linear((((imagew-kernw)//2-kernw//2)//2)**2*2*kernlayers, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



net = Net(kernw = 40)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.001, momentum = 0.9)
#%% Training

printfreq = 20
N = len(train_loader)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = datas

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.type('torch.FloatTensor').reshape(-1,1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print("percentage through data", i/N," loss ",float(sum((outputs- labels.type('torch.FloatTensor').reshape(-1,1))**2)[0]))
        """
        if i % printfreq == printfreq-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            running_loss = 0.0
        """

print('Finished Training')


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