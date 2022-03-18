# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:34:53 2022

@author: Marcu
"""



from scipy.linalg import svd
import numpy as np


N = 100
X = np.zeros((N,30*30))

#Plot variance
# Subtract mean value from data
Y1 = X - np.ones((N,1))*X.mean(axis=0)

#Standardize variation explained
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))

Ys = [Y1, Y2]
k = 1
#U,S,Vh = svd(Ys[k],full_matrices=False)
