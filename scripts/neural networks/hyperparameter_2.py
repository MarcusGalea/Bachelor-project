# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:10:46 2022
Hyperparameter 2.0

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

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import random
from math import floor
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import Dataset, DataLoader, random_split,ConcatDataset,Subset

#%%

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
