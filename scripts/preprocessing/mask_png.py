# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:59:35 2022

@author: aleks
"""
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.io import loadmat
from PIL import Image
import cv2
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

from pathlib import Path
pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)

#%%
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
series = r'AllSeries\\'

img_origin = series + r"CellsCorr_resize\\"
mask_origin = series + r'MaskGT\\'

destination = direc + series + 'mask_png\\'

for mat in os.listdir(direc + mask_origin):

    mask = loadmat(direc + mask_origin + mat)['GTMask']

    try:
        N = mask.shape[2]
    except IndexError:
        N = 1
        
    mask = np.reshape(mask,(mask.shape[0],mask.shape[1],N))
    
    #allocate for mask_im
    im = np.ones((300,300))
    
    for i in range(N):
        mask1 = mask[:,:,i]
        
        txt = mat.split("Image")
        img_name = '_resize' + txt[0][2:] + 'ImageCorr' + txt[1][:-4] + '_mask.png'
        
        mask1 = resize(mask1,(300,300),anti_aliasing=True)
        mask1[mask1 > 0] = 1
        mask1 = mask1.astype(np.uint8)
        
        #temp = (mask1[:,:,i] * (i+1)).astype(np.uint8)
        overlap = im*mask1
        
        im += mask1
    
    im = Image.fromarray(temp)
    #im.save(destination + label + img_name,cmap='gray')
        
        
    break
