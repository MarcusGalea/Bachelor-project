# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:10:10 2022

@author: aleks
"""
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.io import loadmat
import cv2
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#mask_name = r'GT_Serie_2_Image_-1_3992_PC_Cell_Row7_Col_3'
#mask_name2 = r'GT_Serie_2_Image_-4_3990_PC_Cell_Row5_Col_4'
#img_name = r'Serie_2_ImageCorr_-1_3992_PC_Cell_Row7_Col_3.png'

direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
#direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
series = r'AllSeries\\'

img_origin = series + r"CellsCorr_resize\\"
mask_origin = series + r'MaskGT\\'

destination1 = series + r"Crack A\\"
destination2 = series + r"Crack B\\"
destination3 = series + r"Crack C\\"
destination4 = series + r"Finger Failure\\"



def defect_img(img_name,mask_name,new_img_size,direc,series,mask_origin,img_origin,destination):
    #load image and mask
    mask = loadmat(direc + mask_origin + mask_name)['GTMask']
    labels = loadmat(direc + mask_origin + mask_name)['GTLabel']
    img = mpimg.imread(direc+img_origin+img_name)
    try:
        N = mask.shape[2]
    except IndexError:
        N = 1
        
    mask = np.reshape(mask,(mask.shape[0],mask.shape[1],N))
    for i in range(N):
        mask1 = mask[:,:,i]
        label = labels[i][0][0]
        #resize mask
        output_size = img.shape[0]
        mask1 = resize(mask1,(300,300),anti_aliasing=True)
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
        try:
            min_y = min(coords[0])
            
        except ValueError:
            print(mask_name,'contained only zeros.')
            continue
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
        
        #define new coordinates to pull from image.
        new_min_x = center_x - (new_img_size/2)
        new_max_x = center_x + (new_img_size/2)
        new_min_y = center_y - (new_img_size/2)
        new_max_y = center_y + (new_img_size/2)
        
        #check if any goes out of bounds
        if new_min_x < 0:
            new_max_x -= new_min_x
            new_min_x = 0
            
        if new_min_y < 0:
            new_max_y -= new_min_y
            new_min_y = 0
            
        if new_max_x > 300:
            new_min_x -= new_max_x
            new_max_x = 300
            
        if new_max_y > 300:
            new_min_y -= new_max_y
            new_max_y = 300
            
        new_min_x = int(new_min_x)
        new_max_x = int(new_max_x)
        new_min_y = int(new_min_y)
        new_max_y = int(new_max_y)
        
        defect_im = img[new_min_y:new_max_y,new_min_x:new_max_x]
        """
        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(img, cmap='gray')
        ax0.set_title('Original image')
        ax1.imshow(defect_im, cmap='gray')
        ax1.set_title('Defect image')
        plt.show(fig)
        """
        
        mpimg.imsave(direc+destination+ label +"_defect"+ str(i) + img_name, defect_im,cmap = "gray")


k = 0
for mask in os.listdir(direc+mask_origin):
    print(k)
    k +=1
    txt = mask.split("Image")
    
    img_name = '_resize' + txt[0][2:] + 'ImageCorr' + txt[1][:-4] + '.png'
    
    defect_im = defect_img(img_name,mask,50,direc,series,mask_origin,img_origin,destination1)

#%% Sort files into different folders
import shutil

direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
series = r'Full_data\\'

img_origin = series + r"All_images\\"
mask_origin = series + r'All_masks\\'

destination2 = direc + series + r"Crack B\\"
destination3 = direc + series + r"Crack C\\"
destination4 = direc + series + r"Finger Failure\\"


for file_name in os.listdir(direc + series + 'Crack A'):
    if 'Crack B' in file_name:
        source = direc + series + 'Crack A\\' + file_name
        shutil.move(source, destination2)
        print('Moved: ', file_name)
    if 'Crack C' in file_name:
        source = direc + series + 'Crack A\\' + file_name
        shutil.move(source, destination3)
        print('Moved: ', file_name)
    if 'Finger Failure' in file_name:
        source = direc + series + 'Crack A\\' + file_name
        shutil.move(source, destination4)
        print('Moved: ', file_name)