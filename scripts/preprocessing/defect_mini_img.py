# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:10:10 2022

@author: aleks
"""
import os
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
from scipy.io import loadmat
import cv2
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

mask_name = r'GT_Serie_2_Image_-1_3992_PC_Cell_Row7_Col_3'
img_name = r'_noline_Serie_2_ImageCorr_-1_3992_PC_Cell_Row7_Col_3.png'

direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
series = 'Series2\\'

img_origin = series + r"CellsCorr_noline\\"
mask_origin = series + r'MaskGT\\'

destination = series + r"Kernels\\"

#load image and mask
mask = loadmat(direc + mask_origin + mask_name)['GTMask']
img = mpimg.imread(direc+img_origin+img_name)

#resize mask
output_size = 300
mask1 = resize(mask,(300,300),anti_aliasing=True)
mask1[mask1 > 0] = 1
mask1 = mask1.astype(np.uint8)

#show resizing
"""
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(mask, cmap='gray')
ax0.set_title('Boolean array')
ax1.imshow(mask1, cmap='gray')
ax1.set_title('Resized')
plt.show(fig)
"""

#Find max and min x and y coordinates
coords = np.where(mask1 == 1)
min_y = min(coords[0])
max_y = max(coords[0])
min_x = min(coords[1])
max_x = max(coords[1])

center_y = int(sum(coords[0])/len(coords[0]))
center_x = int(sum(coords[1])/len(coords[1]))

"""
fig, ax0 = plt.subplots(1,1)
ax0.imshow(mask1,cmap='gray')
ax0.plot(min_x,min_y,"o")
ax0.plot(max_x,max_y,"o")
ax0.plot(center_x,center_y,"o")
"""

#define new img size, must be an even number. New image will then be NxN pixels
new_size = 50

#define new coordinates to pull from image.
new_min_x = center_x - (new_size/2)
new_max_x = center_x + (new_size/2)
new_min_y = center_y - (new_size/2)
new_max_y = center_y + (new_size/2)

#check if any goes out of bounds
if new_min_x < 0:
    new_max_x -= new_min_x
    new_min_x = 0
    
if new_min_y < 0:
    new_max_y --= new_min_y
    new_min_y = 0
    
if new_max_x > 300:
    new_min_x -= new_max_x
    new_max_x = 300
    
if new_max_y > 300:
    new_min_y -= new_max_y
    new_max_y = 300


 



