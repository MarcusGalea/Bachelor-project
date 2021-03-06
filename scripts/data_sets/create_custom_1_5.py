# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 12:35:12 2022

@author: Marcu
"""
import os
from skimage import io, transform
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%%
import pandas as pd
from math import ceil, floor
from torchvision.io import read_image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split,ConcatDataset,Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, utils
import torchvision
from pytorch_metric_learning.samplers import MPerClassSampler
import random


direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\"
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
series = r"AllSeries\\"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file,delimiter=';',header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels
        label = label.iloc[idx, 1].split('[')[1].split(']')[0].split(',')
        label = np.array(label).astype(float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image[0:1].type(torch.FloatTensor), label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.permute((2,0,1))
        return image

#def load_data(data_dir= None):
data = CustomImageDataset(annotations_file = direc+series+"all_labels2.csv",
                          img_dir = direc+series+r"CellsCorr_resize\\")


labels = data.img_labels.to_numpy()[:,1]


#print(labels)

#seed #DON'T CHANGE, OR ALL PRINCIPAL COMPONENTS MUST BE REMADE
random.seed(10)

#initialize sizes
N = len(data)
train_size = 0.8 #proportion of training data
batch_size = 8 #batch size
m = batch_size//2 #amount of data per class in every batch


#randomly shuffle indices
indices = list(range(N))
random.shuffle(indices)
#create indices for both training data and test data
train_indices = indices[:floor(train_size*N)]
test_indices = indices[floor(train_size*N):]


#split data into training data and test data
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)
train_labels = labels[train_indices]
test_labels = labels[test_indices]

#create sampler for each set of data, s.t each batch contains m of each class
train_sampler = MPerClassSampler(train_labels, m, batch_size=batch_size, length_before_new_iter=100000)
#test_sampler = MPerClassSampler(test_labels, m, batch_size=batch_size, length_before_new_iter=100000)

#%% create dataloaders
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}



#dataloader for training data
train_loader = DataLoader(
    train_split,
    shuffle=False,
    sampler = train_sampler,
    batch_size=batch_size,
    **kwargs
)

#dataloader for test data
test_loader = DataLoader(
    test_split,
    shuffle=False,
    batch_size=batch_size,
    **kwargs
)
    #return train_loader, test_loader

#train_loader, test_loader = load_data(direc+series)
#%%
for i,datas in enumerate(train_loader):
    inputs,labels = datas
    break