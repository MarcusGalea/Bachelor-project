# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:10:46 2022
Hyperparameter 2.0

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
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
#os.chdir(dname)


pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)


# %%

global datadir
global labels
global images
global avg_im

user = "HPC"
if user == "Aleksander":
    direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\"
    series = r"AllSeries\\"
    images = r"CelssCorr_resize\\"
    labels = r"all_labels.csv"
    
    
elif user == "Marcus":
    direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
    series = r"AllSeries\\"
    images = "CellsCorr_resize"
    labels = "labels.csv"
    
elif user == "HPC":
    direc = '/zhome/35/5/147366/Desktop/'
    series = ''
    images = 'CellsCorr_resize'
    labels = 'labels.csv'


datadir = direc+series
avg_im = read_image(datadir +"_average_cell.png")[0]

#from data_sets.create_custom_1 import CustomImageDataset
# %%
 
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

# %%
def load_data(data_dir = datadir,labels = labels,images = images, sample_test = False):
    data = CustomImageDataset(annotations_file = data_dir + labels,img_dir = data_dir + images,transform = transforms.RandomVerticalFlip())
    
    
    labels = data.img_labels.to_numpy()[:,1]
    
    
    
    #seed #DON'T CHANGE, OR ALL PRINCIPAL COMPONENTS MUST BE REMADE
    #random.seed(10)
    
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
    if sample_test:
        test_sampler = MPerClassSampler(test_labels, m, batch_size=batch_size, length_before_new_iter=1000)
    else:
        test_sampler = None
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        sampler = test_sampler,
        batch_size=batch_size,
        **kwargs
    )
    return train_loader, test_loader

#train_loader, test_loader = load_data(direc+series)
# %%


class Net(nn.Module):
    def __init__(self,
                 kernw=30,  # width/height of convolution
                 # number number of layers in first convolution (twice as many in second convolution)
                 kernlayers=6,
                 l1=120,  # number of outputs in first linear transformation
                 l2=84,  # number of outputs in second linear transformation
                 imagew=400,  # width/height of input image
                 drop_p = 0.5 # dropout rate
                 ):
        super().__init__()
        # third arg: remove n-1 from img dim
        self.conv1 = nn.Conv2d(1, kernlayers, kernw)
        self.pool = nn.MaxPool2d(2, 2)  # divide img dim with n
        self.conv2 = nn.Conv2d(kernlayers, 2*kernlayers, kernw//2)
        self.fc1 = nn.Linear(
            (((imagew-kernw)//2-kernw//2)//2)**2*2*kernlayers, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 2)
        self.drop = nn.Dropout(drop_p)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
def train_cifar(config, checkpoint_dir = None, data_dir = None,labels = None,images = None):
    net = Net(kernw=config["kernw"],kernlayers = config["kernlayers"], l1=config["l1"], l2=config["l2"],drop_p = config["dropout"])
    
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    
    w = torch.tensor(config["weight"])
    if device == "cuda:0":
        w = w.type(torch.cuda.FloatTensor)#.to(device)
    
    criterion = nn.CrossEntropyLoss(weight = w)
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader, valloader = load_data(data_dir,labels,images,sample_test = True)

    printfreq = 20
    for epoch in range(10):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, datas in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            #remove average image to remove lines
            inputs, labels = datas #range = (0,250)
            inputs -= avg_im #range = (-250,250)
            inputs /= 255 #range = (-1,1)
            inputs += 1 # range = (0,2)
            inputs /= 2 # range = (0,1)
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
                running_loss = 0.0
    
            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs -= avg_im #range = (-250,250)
                    inputs /= 255 #range = (-1,1)
                    inputs += 1 # range = (0,2)
                    inputs /= 2 # range = (0,1)
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
    

# %%
def test_accuracy(net, device="cpu",data_dir = datadir,labels = labels,images = images):
    trainloader, testloader = load_data(data_dir, labels, images)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images -= avg_im #range(-250,250)
            images /= 255 #range(-1,1)
            images += 1 #range(0,2)
            images /= 2 #range(0,1)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
# %%


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):

    load_data(datadir, labels,images)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "kernw": tune.choice([40, 50, 60, 70, 80, 90]),
        "kernlayers": tune.choice([4, 6, 8, 10, 12, 14]),
        "weight": tune.choice([[1.,10.],[1.,20.],[1.,30.],[1.,40.]]),
        "dropout": tune.choice([0.4,0.5,0.6])
        # "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "lr", "kernw","kernlayers","weight","dropout"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar,checkpoint_dir = dname,data_dir = direc+series,labels=labels,images=images),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
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

    best_trained_model = Net(l1 = best_trial.config["l1"], l2 = best_trial.config["l2"],kernw = best_trial.config["kernw"],kernlayers = best_trial.config["kernlayers"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    #torch.save(best_trained_model.state_dict(),"NN_1_7.pt")

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=250, max_num_epochs=10, gpus_per_trial=1)
