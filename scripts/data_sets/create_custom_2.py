# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:40:26 2022

@author: aleks
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
import matplotlib.image as mpimg
from scipy.io import loadmat
from skimage.transform import resize
import utils

#%%
#direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"
series = r"AllSeries\\"

global mask1
class CustomImageDataset2(Dataset):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "CellsCorr_faulty"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "MaskGT"))))

    def __getitem__(self, idx):
        
        N_im = 400
        
        # load images and masks
        img_path = os.path.join(self.root, "CellsCorr_faulty", self.imgs[idx])
        mask_path = os.path.join(self.root, "MaskGT", self.masks[idx])
        img = mpimg.imread(img_path)[:,:,0]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = loadmat(mask_path)
        temp_label = mask['GTLabel']             
        mask = mask['GTMask']
        
        num_labels = len(temp_label)
        try:
            num_objs = mask.shape[2]
        except IndexError:
            num_objs = 1
            
        mask = np.reshape(mask,(mask.shape[0],mask.shape[1],num_objs))
        iscrowd = np.zeros((num_labels,))
        
        masks = np.zeros((N_im,N_im))
        labels = []
        boxes = []
        for i in range(num_labels):
            ## labels
            ID = temp_label[i][0][0]
            if ID == 'Crack A':
                labels.append(1)
            if ID == 'Crack B':
                labels.append(2)
            if ID == 'Crack C':
                labels.append(3)
            if ID == 'Finger Failure':
                labels.append(4)
        if len(labels)<1:
            labels = [0]
        
        
        k = 0
        for i in range(num_objs):
            mask1 = mask[:,:,i]
            if not(mask1==1).any():
                continue

            k += 1
                
            #resize mask
            mask1 = resize(mask1,(N_im,N_im),anti_aliasing=True)
            mask1[mask1 > 0] = 1
            masks[mask1 > 0] = k+1

        # get bounding box coordinates for each mask
            pos = np.where(mask1==1)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list,label_list
    
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = CustomImageDataset2(direc + series)


data_loader = torch.utils.data.DataLoader(
 dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn = collate_fn)


#%%
N = len(data_loader)
for i, data in enumerate(data_loader):
    img,target = data
    print(i/N*100)


#%%
N_im = 300
im1 = "GT_Serie_2_Image_-12_4009_PC_Cell_Row6_Col_4.mat"
im2 = "GT_Serie_1_Image_-16_4133_Cell_Row7_Col_2.mat"
mask_path = direc + series + r"MaskGT\\"+ im2

mask = loadmat(mask_path)
temp_label = mask["GTLabel"]
mask = mask["GTMask"]


num_labels = len(temp_label)
try:
    num_objs = mask.shape[2]
except IndexError:
    num_objs = 1
    
mask = np.reshape(mask,(mask.shape[0],mask.shape[1],num_objs))
iscrowd = np.zeros((num_labels,))

masks = np.zeros((N_im,N_im))
labels = []
boxes = []
for i in range(num_labels):
    ## labels
    ID = temp_label[i][0][0]
    if ID == 'Crack A':
        labels.append(1)
    if ID == 'Crack B':
        labels.append(2)
    if ID == 'Crack C':
        labels.append(3)
    if ID == 'Finger Failure':
        labels.append(4)
if len(labels)<1:
    labels = [0]


k = 0
for i in range(num_objs):
    mask1 = mask[:,:,i]
    plt.imshow(mask1,cmap='gray')
    plt.title("original")
    plt.show()
    if not(mask1==1).any():
        continue

    k += 1
        
    #resize mask
    mask1 = resize(mask1,(N_im,N_im),anti_aliasing=True)
    mask1[mask1 > 0] = 1
    masks[mask1 > 0] = k+1
    plt.imshow(mask1,cmap="gray")
    plt.title("scaled")
    plt.show()

# get bounding box coordinates for each mask
    pos = np.where(mask1==1)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    boxes.append([xmin, ymin, xmax, ymax])

# convert everything into a torch.Tensor
boxes = torch.as_tensor(boxes, dtype=torch.float32)
labels = torch.as_tensor(labels, dtype=torch.int64)
masks = torch.as_tensor(masks, dtype=torch.uint8)


area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

target = {}
target["boxes"] = boxes
target["labels"] = labels
target["masks"] = masks
target["area"] = area
target["iscrowd"] = iscrowd