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


#%%

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
series = r"AllSeries\\"

#%%
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image[0:1].type(torch.FloatTensor), label


#%%
def load_data(data_dir= None):
    data = CustomImageDataset(annotations_file = direc+series+"labels.csv",
                              img_dir = direc+series+r"CellsCorr_noline\\")
    
    labels = data.img_labels.to_numpy()[:,1]
    
    #%% Big dataset. Concatenate all data to the same dataset
    ## Since creation of "AllSeries" folder, this part has become obsolete
    """
    #choose series to include in data
    serieslist = [r"Series3\\",r"Series4\\",r"Series6\\"]
    datasets = [data]
    
    
    for series in serieslist:
        #load data from series
        newdata = CustomImageDataset(annotations_file = direc+series+"labels.csv",
                                  img_dir = direc+series+r"CellsCorr_noline\\")
        #load labels from series
        newlabels = newdata.img_labels.to_numpy()[:,1]
        
        #concatenate data and labels to big dataset
        labels = np.concatenate((labels,newlabels))
        datasets.append(newdata)
    data = ConcatDataset(datasets)
    
    #%% split data
    """
    
    
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
    train_sampler = MPerClassSampler(train_labels, m, batch_size=batch_size, length_before_new_iter=10000)
    test_sampler = MPerClassSampler(test_labels, m, batch_size=batch_size, length_before_new_iter=10000)
    
    #%% create dataloaders
    
    
    #dataloader for training data
    train_loader = DataLoader(
        train_split,
        shuffle=False,
        num_workers=0,
        sampler = train_sampler,
        batch_size=batch_size
    )
    
    #dataloader for test data
    test_loader = DataLoader(
        test_split,
        shuffle=False,
        num_workers=0,
        sampler = test_sampler,
        batch_size=batch_size
    )
    return train_loader, test_loader
#%%
class Net(nn.Module):
    def __init__(self,
                 kernw = 30, #width/height of convolution
                 kernlayers = 6, #number number of layers in first convolution (twice as many in second convolution)
                 l1 = 120, #number of outputs in first linear transformation
                 l2 = 84, #number of outputs in second linear transformation
                 imagew = 300 #width/height of input image
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, kernlayers, kernw) #third arg: remove n-1 from img dim
        self.pool = nn.MaxPool2d(2, 2) #divide img dim with n
        self.conv2 = nn.Conv2d(kernlayers, 2*kernlayers, kernw//2)
        self.fc1 = nn.Linear((((imagew-kernw)//2-kernw//2)//2)**2*2*kernlayers, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



#%%
def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader, valloader = load_data(data_dir)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")
    
    
#%%
def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
#%% 
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = direc+series
    load_data(data_dir)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)