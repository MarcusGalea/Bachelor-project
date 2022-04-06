# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:04:52 2022

This scripts now reshapes images and saves them if avg pixel is above certain threshold

@author: Marcu
"""
import os
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#direc = r"C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\"
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"

series = r"AllSeries\\"
origin = series + r"CellsCorr\\"

destination = series + r"CellsCorr_resize\\"



import sys
sys.path.insert(1, direc+'scripts')
from functions import *
import cv2 as cv
import matplotlib.pyplot as plt
#%%
k = 0
N = len(os.listdir(direc+origin))
n = 400
m = n
avg_im = np.zeros((n,m))

imlist = os.listdir(direc+origin)
for pic in imlist[k:]:
    
    k +=1
    print(k)
    if pic != "Thumbs.db":
        img = mpimg.imread(direc+origin+pic)
        img = resize(img, (n, m))
        avg_im += img/N
        
        avg = np.mean(img)
        
        if avg > 0.28:
            mpimg.imsave(direc+destination+"_resize_"+pic, img,cmap = "gray")



#%%
n =400
m = n
img = mpimg.imread(direc+series+"_average_cell.png")
img = resize(img, (n, m))
mpimg.imsave(direc+series+"_average_cell.png", img,cmap = "gray")

        
        