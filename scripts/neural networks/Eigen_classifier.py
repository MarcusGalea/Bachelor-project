# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:27:32 2022

@author: Marcu
"""

import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"

series = r"AllSeries\\"
kernel = r"Kernels\PC\\"
defects = [r"Crack A\\",r"Crack B\\",r"Crack C\\", r"Finger Failure\\"]
PC_link = direc+series+kernel


weights = np.zeros((4,10,50,50))
avg = np.zeros((4,50,50))


for i,defect in enumerate(defects):
    k = -1
    for PC in os.listdir(PC_link+defect):
        PC = mpimg.imread(PC_link+defect + PC)
        PC = PC[0:1]
        PC /= np.linalg.norm(PC)
        if k == -1:
            k+= 1
            avg[i] = PC
            continue
        weights[i][k] = PC
        k+= 1