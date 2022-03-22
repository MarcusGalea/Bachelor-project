# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:17:46 2022

@author: aleks
"""

import os
import numpy as np
from scipy.linalg import svd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\Full_data\\'
direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"

series = r"AllSeries\Kernels\\"
defect = r"Crack C\\"

#initialize array
A = np.array([])

for kernel in os.listdir(direc+series+defect):
    img = mpimg.imread(direc+series + defect+kernel)[:,:,0]
    
    temp = img.flatten()
    temp = temp.reshape(-1,1)
    
    if A.size == 0:
        A = temp
    
    A = np.hstack((A,temp))

#%%
N,M = A.shape
mean = A.mean(axis=1)
Y = A - np.tile(mean.reshape(-1,1),(1,M))
M = Y@Y.T

U,S,Vh = svd(M,full_matrices=False)

#%%
rho = (S*S) / (S*S).sum() 
n = 10

plt.plot(np.arange(0,n+1),np.hstack(([0],rho.cumsum()[:10])))
# Compute the projection onto the principal components
Z = U*S;


#%%
import matplotlib.image as mpimg
from matplotlib.pyplot import cm

N_k = int(np.sqrt(N))


destination = r'PC\\'
mpimg.imsave(direc + series +destination + defect + 'avg.png',mean.reshape(N_k,N_k),cmap=cm.gray)
#save the n first kernels
for i in range(10):
    img = Vh[i,:]
    img = img.reshape(N_k,N_k)
    mpimg.imsave(direc + series +destination + defect + 'kernel'+str(i)+'.png',img,cmap=cm.gray)


#%%
gamma = A[:,0]
im1 = gamma.reshape(50,50)
plt.imshow(im1,cmap = "gray")

Omega = gamma - mean

N_PC = 10
omega = np.zeros(N_PC)
proj = np.zeros((50,50))

for i in range(N_PC):
    omega[i] = np.dot(Omega,Vh[i,:])
    proj += omega[i]*Vh[i,:].reshape(50,50)
    
#%%
plt.imshow(proj,cmap = "gray")
error = np.linalg.norm(im1-proj)
print(error)

