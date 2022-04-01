# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:49:47 2022

@author: Marcu
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import shutil


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

pathname = Path(dname)
pathname = pathname.parent.absolute()
direc = pathname.parent.absolute()
os.chdir(direc)
#%%

k = 0
#direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"
series = r"AllSeries\\"
images = direc + series + r"CellsCorr_resize\\"
faulty_images = direc + series + r"CellsCorr_faulty\\"
labels = direc + series + r"MaskGT\\"

y = []
dic = {}
k = 0

n = 10
m = 6
for pic in os.listdir(images):
    if pic != "Thumbs.db":
        serie = pic.split("_")[3]
        txt = pic.split("Corr")[1]
        txt = txt.split(".")[0]
        dic[serie+txt] = k
        y.append([pic,0])
        k += 1

for label in os.listdir(labels):
    mask = loadmat(labels + label)['GTMask']
    
    try:
        N = mask.shape[2]
    except IndexError:
        N = 1
        
    mask = np.reshape(mask,(mask.shape[0],mask.shape[1],N))
    
    if sum(sum(sum(mask))) == 0:
        print(label, 'discarded')
        continue
    
    serie = label.split("_")[2]
    txt = label.split("Image")[1]
    txt = txt.split(".")[0]
    try:
        y[dic[serie+txt]][1] = 1
        im_title = "_resize_Serie_" +serie+ "_ImageCorr"+txt+".png"
        shutil.copyfile(images+im_title, faulty_images+im_title)
    except KeyError:
        print("cells for "+txt+" are missing")
        continue

pd.DataFrame(y).to_csv(direc + series + "labels.csv",header = None, index = None)
