"""
Authors:
    Aleksander Svendstorp - s194323
    Marcus Galea Jacobsen - s194336
    
Last edited: 19/05-2022

Script name: create_csv.py
Script function: Script that creates two csv files. One csv file (binary labels) will be used for the custom dataset, the other (multi class labels) will be used for creating class statistics 
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


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

pathname = Path(dname)
pathname = pathname.parent.absolute()
direc = pathname.parent.absolute()
os.chdir(direc)
#%%

k = 0
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"
series = r"AllSeries\\"
images = direc + series + r"CellsCorr_resize300\\"
faulty_images = direc + series + r"CellsCorr_faulty\\"
labels = direc + series + r"MaskGT\\"

y = []
y2 = []
dic = {}
k = 0
avg = []

n = 10
m = 6
h = 0
min_avg = 1000
for pic in os.listdir(images):
    if pic != "Thumbs.db":
        serie = pic.split("_")[3]
        txt = pic.split("Corr")[1]
        txt = txt.split(".")[0]
        dic[serie+txt] = k
        y.append([pic,0])
        k += 1

for label in os.listdir(labels):
    GT = loadmat(labels + label)
    mask1 = GT['GTMask']
    mat_label = GT['GTLabel']

    try:
        N = mask1.shape[2]
    except IndexError:
        N = 1

    mask = np.reshape(mask1,(mask1.shape[0],mask1.shape[1],N))
    mask1d = np.zeros((mask1.shape[0],mask1.shape[1],1))
    temp = np.copy(mask)
    temp[temp > 0.5] = 1
    
    helper = False
    
    serie = label.split("_")[2]
    txt = label.split("Image")[1]
    txt = txt.split(".")[0]
    im_title = "_resize_"+"Serie_" +serie+ "_ImageCorr"+txt+".png"
    
    if not(mask == 1).any():
        print(label, 'discarded')
        continue
    for i in range(N):
        idx = np.where(temp[:,:,i] == 1)
        mask1d[idx] = i+1
        if sum(sum(temp[:,:,i])) > 200:
            helper = True
    if helper:
        try:
            y[dic[serie+txt]][1] = 1
            for i in range(N):
                if mat_label[i][0][0] == 'Finger Failure':
                   y2[dic[serie+txt]][1][3]  = 1
                if mat_label[i][0][0] == 'Crack A':
                    y2[dic[serie+txt]][1][0] = 1
                if mat_label[i][0][0] == 'Crack B':
                    y2[dic[serie+txt]][1][1] = 1
                if mat_label[i][0][0] == 'Crack C':
                    y2[dic[serie+txt]][1][2] = 1
        except KeyError:
            print("cells for "+txt+" are missing")
            continue

pd.DataFrame(y).to_csv(direc + series + "labels.csv",header = None, index = None)
pd.DataFrame(y2).to_csv(direc + series + "labels_MC.csv",header = None, index = None)
