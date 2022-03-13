#Get pictures

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import numpy as np
import os

# Display PNG file
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Bachelor-project--defect-detection-on-solar-panels"


#%% show image
img = mpimg.imread(direc + 'data/CellsFrSerie1_10_4081/Serie_1_ImageGS_-10_4081_Cell_Row1_Col_1.png')

plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram
threshold = 0.3
plt.show()


binimg = img.copy()
background = np.where(img<threshold)
black = np.where(img>=threshold)
binimg[background] = 0
binimg[black] = 1


imgnm = plt.imshow(img,cmap='gray')

x_axis = imgnm.axes.get_xaxis()
#x_axis.set_visible(False)

y_axis = imgnm.axes.get_yaxis()
#y_axis.set_visible(False)

plt.show()


#%% Blob-analysis
import sys
sys.path.insert(1, direc+'scripts')
from functions import *

N = 12
tol = 0.1
p = 0.2
N = 6
tol = 0.5
#mask = grassfire(img,N,p)
lines = find_lines(binimg,tol)
img = remove_lines(img,lines)

#%% 
imgnm = plt.imshow(img,cmap='gray')

x_axis = imgnm.axes.get_xaxis()
x_axis.set_visible(False)

y_axis = imgnm.axes.get_yaxis()
y_axis.set_visible(False)

plt.show()
#%%
N = 12
p = 0.2
mask = grassfire(img,N,p)


#%% 

plt.show()
threshold = 0.4
n,m = np.shape(img)
binimg = np.zeros((n,m))
background = np.where(img<threshold)
black = np.where(img>=threshold)
binimg[background] = 0
binimg[black] = 1



#%% 
imgnm = plt.imshow(mask,cmap='gray')

x_axis = imgnm.axes.get_xaxis()
x_axis.set_visible(False)

y_axis = imgnm.axes.get_yaxis()
y_axis.set_visible(False)

plt.show()
