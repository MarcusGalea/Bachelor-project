# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:55:43 2022

@author: Marcu
"""

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torchvision.io import read_image
from math import ceil, floor
import pandas as pd
from pytorch_metric_learning.samplers import MPerClassSampler
from math import floor
import random
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from functools import partial
import os
from pathlib import Path
import matplotlib.image as mpimg
from scipy.io import loadmat
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
#os.chdir(dname)


pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)
#%%
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

global datadir
global labels
global images
global avg_im

user = "Marcus"
if user == "Aleksander":
    direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\AllSeries\\"
    series = r"AllSeries\\"
    images = r"CellsCorr\\"
    labels = r"all_labels.csv"
    
    
elif user == "Marcus":
    direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
    series = r"AllSeries\\"
    direc += series
    images = "CellsCorr_resize"
    labels = "labels.csv"
    
elif user == "HPC":
    direc = '/zhome/35/5/147366/Desktop/'
    series = ''
    images = 'CellsCorr_resize300'
    labels = 'labels300.csv'


#%%
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
        
        #N_im = 400
        
        # load images and masks
        img_path = os.path.join(self.root, "CellsCorr_faulty", self.imgs[idx])
        mask_path = os.path.join(self.root, "MaskGT", self.masks[idx])
        img = mpimg.imread(img_path)
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
        
            
        mask = np.reshape(mask,(mask.shape[0],mask.shape[1],num_labels))
        iscrowd = []
        
        masks = np.zeros((mask.shape[0],mask.shape[1],1))
        labels = []
        boxes = []        
        
        for i in range(num_labels):
            mask1 = mask[:,:,i]
            if not(mask1==i+1).any():
                continue
            if sum(sum(mask1))/(i+1) < 200:
                iscrowd.append(1)
            else:
                iscrowd.append(0)
            
            ID = temp_label[i][0][0]
            if ID == 'Crack A':
                labels.append(1)
            if ID == 'Crack B':
                labels.append(2)
            if ID == 'Crack C':
                labels.append(3)
            if ID == 'Finger Failure':
                labels.append(4)
                
            #resize mask
            #mask1 = resize(mask1,(N_im,N_im),anti_aliasing=True)
            #mask1[mask1 > 0.5] = 1
            #masks[mask1 > 0.5] = k+1

        # get bounding box coordinates for each mask
            pos = np.where(mask1==i+1)
            masks[pos] = 1
            
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1
            boxes.append([xmin, ymin, xmax, ymax])
        
        if len(labels)<1:
            labels = [0]
            n,m = np.shape(img)
            boxes.append([0,0,m,n])
            area = [n*m]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        area = torch.as_tensor(area,dtype=torch.int64)
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

#%% Training
#from engine import train_one_epoch, evaluate
from torchvision import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


#Faster R-CNN
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
"""
# load a pre-trained model for classification and return
# only the features
#backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes = 5).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((16, 64, 256),),
                                   aspect_ratios=((0.1, 1.0, 3.0, 6.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=5,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
"""
#mask R-CNN

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model

#import transforms as T 
def get_transform(train):
    transforms = []
    #transforms.append(T.ToTensor())
    if train:
        0
        #transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(transforms.RandomVerticalFlip(0.5))
    return 0#T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 5
    # use our dataset and defined transformations
    #dataset = CustomImageDataset2(direc, get_transform(train=False))
    dataset_test = CustomImageDataset2(direc)#, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(0)
    indices = torch.randperm(len(dataset_test)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
    """
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)
    """
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    # get the model using our helper function
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes = num_classes)
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,num_classes = num_classes)
    model = get_model_instance_segmentation(num_classes)
    
    # move model to the right device
    model.to(device)



        
    PATH = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\\NN_2_15.pt"
   #torch.load(model.state_dict(), PATH)
    
    # evaluate on the test dataset
    #evaluate(model, data_loader_test, device=device)
    return model, data_loader_test
    
model, data_loader_test = main()

#%%
images,targets = next(iter(data_loader_test))

images = (images[0:1])
targets = (targets[0:1])
images = list(image for image in images)
images = torch.tensor(images)
images
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)
