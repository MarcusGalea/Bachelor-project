# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:30:30 2022

@author: Marcu
"""


import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg

from scipy.io import loadmat
import shutil
import matplotlib.pyplot as plt
import time
import torch


#%%

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%%
from NN import Net
#%%


from create_custom_1 import train_loader,test_split,train_sampler,train_split

from skorch import NeuralNetClassifier,NeuralNetBinaryClassifier
from torch.utils.data import Subset,DataLoader
#%%
from pytorch_metric_learning.samplers import MPerClassSampler
from skorch.helper import SliceDataset

#%%
y_train_np = np.array([y for x,y in iter(train_split)])
y_test_np = np.array([y for x,y in iter(test_split)])

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
m = batch_size//2
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}



n = len(train_split)//2
y_cal = y_train_np[n:]
y_train = y_train_np[:n]
x_cal = Subset(train_split, np.arange(n))
x_train = Subset(train_split, np.arange(n,n*2))


#create sampler for each set of data, s.t each batch contains m of each class
train_sampler = MPerClassSampler(y_train, 
                                 m, 
                                 batch_size=batch_size, 
                                 length_before_new_iter=100000)



train_loader = DataLoader(
    x_train,
    shuffle=False,
    #sampler = train_sampler,
    batch_size=batch_size,
    **kwargs
)
cal_loader = DataLoader(
    x_cal,
    shuffle=False,
    #sampler = train_sampler,
    batch_size=batch_size,
    **kwargs
)


#dataloader for test data
test_loader = DataLoader(
    test_split,
    shuffle=False,
    #sampler = test_sampler,
    batch_size=batch_size,
    **kwargs
)
#%%
import pickle

w = torch.tensor([1.,15.])
if device == "cuda:0":
    w = w.type(torch.cuda.FloatTensor)#.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=w)

net = Net()

"""
PATH = "NN_1_5.pt"

if device == "cuda:0":
    net.load_state_dict(torch.load(PATH))
elif device == "cpu":
    net.load_state_dict(torch.load(PATH,map_location = torch.device('cpu')))

net.eval()
"""
net_m = NeuralNetClassifier(net,
                        criterion = criterion, 
                        max_epochs = 10, 
                        lr = 0.0001, 
                        batch_size = 8, 
                        optimizer = torch.optim.Adam,
                        dataset = x_train,
                        train_split = None,
                        iterator_train__sampler = train_sampler,
                        #iterator_train = train_loader,
                        device = device
                        )

"""
net_m.initialize()  # This is important!
net_m.load_params(f_params=PATH)
"""

ds = SliceDataset(x_train)
net_m.fit(X = x_train,y=y_train)

with open('net_m_1_6.pkl', 'wb') as f:
    pickle.dump(net_m, f)
#%%
"""
with open('net_m_1_5.pkl', 'rb') as f:
    net_m = pickle.load(f)

from mapie.classification import MapieClassifier

net_m.module.eval()

#small_test = Subset(test_split,np.arange(1000))
small_test = test_split

y_pred = net_m.predict(X = small_test)

TP = np.intersect1d(np.where(y_pred==1)[0],np.where(y_test_np==1)[0])
#%%
cal_arr = np.array([])
for i in range(len(x_cal)):
    
#%%
mapie_score = MapieClassifier(estimator=net_m, cv="prefit", method="score")
mapie_score.fit(x_cal, y_cal)
alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(x_cal, alpha=alpha)
"""