"""
Authors:
    Aleksander Svendstorp - s194323
    Marcus Galea Jacobsen - s194336
    
Last edited: 19/05-2022

Script name: discard_PID.py
Script function: Script that discards images with average less than 0.28, with the intention of discarding images with PID defects
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
series = r"AllSeries\\"
images = direc + series + r"CellsCorr_resize\\"
masks = direc + series + r"MaskGT\\"

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
        
        if not((mask == 1).any()):
            continue
    except FileNotFoundError:
        if avg < 0.28:
            os.remove(images + pic)
    
        
        

