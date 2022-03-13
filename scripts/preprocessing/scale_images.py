# -*- coding: utf-8 -*-
## Authors:
# Aleksander Svendstorp - s194323
# Marcus Galea Jacobsen - s194336
# Last edited: 28-02-2022

## scale_images.py
# Scales images to the same size of n x m.


import os
import matplotlib.image as mpimg
import math
import cv2
import numpy as np

folder = r'C:\Users\aleks\OneDrive\Dokumenter\GitHub\Bachelor-project--defect-detection-on-solar-panels\data\Series6\CellsCorr\\'

destination = r'C:\Users\aleks\OneDrive\Dokumenter\GitHub\Bachelor-project--defect-detection-on-solar-panels\data\Series6\CellsCorr_resize2\\'

n = 300
m = 300

count = 0

for pic in os.listdir(folder):
    if pic != 'Thumbs.db':
        img = mpimg.imread(folder + pic)
        new_img = cv2.resize(img, (n, m))
        
        mpimg.imsave(destination + '_resized_' + pic, new_img,cmap='gray')
        
        count += 1
        print('Image',count,'resized and saved')
    
print(count,'images resized and saved')