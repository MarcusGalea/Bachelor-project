# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:45:27 2022

@author: Marcu
"""

import os
from pathlib import Path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

#pathname = Path(dname)
#parent_folder = pathname.parent.absolute()
os.chdir(dname)
from create_custom_1 import *
from NN import *


#%% Training


printfreq = 100
N = len(train_loader)


table = []
for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, datas in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        #remove average image to remove lines
        inputs, labels = datas #range = (0,250)
        #inputs -= avg_im #range = (-250,250)
        inputs /= 255 #range = (-1,1)
        #inputs += 1 # range = (0,2)
        #inputs /= 2 # range = (0,1)
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
        
        running_loss += loss.item()
        # print statistics
        
        if i % printfreq == printfreq-1:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printfreq:.3f}')
            table.append([epoch +1, i+1, running_loss / printfreq])
            running_loss = 0.0

print('Finished Training')
PATH = "NN_1_10_HP_1.pt"
torch.save(net.state_dict(), PATH)
pd.DataFrame(table).to_csv(dname + "\\loss_1_10_HP_1.csv",header = None, index = None)

