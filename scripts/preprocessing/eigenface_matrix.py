# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:17:46 2022

@author: aleks
"""

import os
import numpy as np
from scipy.linalg import svd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\Full_data\\'
series = r'Crack A\\'

#initialize array
A = np.array([])

for kernel in os.listdir(direc + series):
    img = mpimg.imread(direc+series + kernel)[:,:,0]
    
    temp = img.flatten()
    temp = temp.reshape(-1,1)
    
    if A.size == 0:
        A = temp
    
    A = np.hstack((A,temp))
   
N = A.shape[0]
Y = A - np.ones((N,1))*A.mean(axis=0)

M = Y@Y.T

U,S,Vh = svd(M,full_matrices=False)

#%%
import matplotlib.image as mpimg
from matplotlib.pyplot import cm

destination = r'Kernels\\'

#save the n first kernels
for i in range(10):
    
    img = Vh[i,:]
    img = img.reshape(50,50)
    mpimg.imsave(direc + series + destination + 'kernel'+str(i)+'.png',img,cmap=cm.gray)
