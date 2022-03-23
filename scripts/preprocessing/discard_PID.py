# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:37:24 2022

@author: aleks
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%
direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\data\\"
series = r"AllSeries\\"
images = direc + series + r"CellsCorr_resize\\"

avg_vec = []

for pic in os.listdir(images):
    
    img = mpimg.imread(images + pic)[:,:,0]
    
    avg = np.mean(img)
    
    avg_vec.append(avg)
    
    #if avg < 0.2:
        #print(pic)
        #plt.imshow(img)
