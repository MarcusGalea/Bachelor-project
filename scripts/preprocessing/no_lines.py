# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:04:52 2022

Removes vertical lines from data

@author: Marcu
"""
import os
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize



direc = r"C:\Users\aleks\OneDrive\Dokumenter\GitHub\Bachelor-project--defect-detection-on-solar-panels\data\\"
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Bachelor-project--defect-detection-on-solar-panels\data\\"

series = r"Series4\\"
origin = series + r"CellsCorr\\"

destination = series + r"CellsCorr_noline\\"


import sys
sys.path.insert(1, direc+'scripts')
from preprocessing.functions import *
import matplotlib.pyplot as plt

k = 0
for pic in os.listdir(direc+origin):
    print(k)
    k +=1
    if pic != "Thumbs.db":
        img = mpimg.imread(direc+origin+pic)
        n,m = np.shape(img)
        binimg = np.zeros((n,m))
        threshold = np.quantile(img,q=0.30)
        
        background = np.where(img<threshold)
        black = np.where(img>=threshold)
        binimg[background] = 0
        binimg[black] = 1
        
        tol = 0.3
        lines = find_lines(binimg,tol)
        img = remove_lines(img,lines)
        
        img = resize(img, (250, 250))
        
        mpimg.imsave(direc+destination+"_noline_"+pic, img,cmap = "gray")
        
        