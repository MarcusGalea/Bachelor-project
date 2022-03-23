# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:37:24 2022

@author: aleks
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.transform import resize

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%
direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"
series = r"AllSeries\\"
images = direc + series + r"CellsCorr_resize\\"
masks = direc + series + r"MaskGT\\"

#avg_vec = []

#mean and std for all images avg
mu = 0.54576164
sigma = 0.098812304

for pic in os.listdir(images):
    
    img = mpimg.imread(images + pic)[:,:,0]
    
    avg = np.mean(img)
    
    txt = pic.split('resize_')
    txt = txt[1].split('.')
    txt = 'GT_' + txt[0]
    
    try:
        mask = loadmat(masks + txt)['GTMask']
        try:
            N = mask.shape[2]
        except IndexError:
            N = 1
        
        mask = np.reshape(mask,(mask.shape[0],mask.shape[1],N))
        
        if sum(sum(sum(mask))) != 0:
            continue
    except FileNotFoundError:
        if avg < 0.2:
            #print(pic)
            plt.imshow(img,cmap = 'gray')
            plt.show()
    
    #avg_vec.append(avg)
        
        

