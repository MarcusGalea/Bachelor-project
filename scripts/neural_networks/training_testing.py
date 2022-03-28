# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:45:27 2022

@author: Marcu
"""

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(dname)
from NN import *


#%% Training

#import multiprocessing as mp
#import torch.multiprocessing as mp_t

printfreq = 20
N = len(train_loader)

table = []
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = datas
        inputs /= 255
        #inputs -= avg_im
        if device == "cuda:0":
            inputs = inputs.type(torch.cuda.FloatTensor)#.to(device)
            labels = labels.type(torch.cuda.LongTensor)
        #labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % printfreq == printfreq-1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            table.append([epoch +1, i+1, running_loss / printfreq])
            running_loss = 0.0

print('Finished Training')

PATH = "NN_1_2.pt"
torch.save(net.state_dict(), PATH)

