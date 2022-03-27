# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:27:32 2022

@author: Marcu
"""

import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from itertools import islice
from csv import reader

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from pathlib import Path
pathname = Path(dname)
parent_folder = pathname.parent.absolute()
os.chdir(parent_folder)

from data_sets.create_custom_1 import test_loader

#%%

direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Data\\"
#direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\Full_data\\'


series = r"AllSeries\Kernels\\"

destination = r'PC\\'

defects = [r"Crack A\\",r"Crack B\\",r"Crack C\\", r"Finger Failure\\"]



N_PCs = 100
PCs = np.zeros((4,N_PCs,50*50))


for i,defect in enumerate(defects):
    defect = defect.split("\\")[0].split(" ")[0]+"_"+defect.split("\\")[0].split(" ")[1]
    with open(direc+series+destination+defect+"_PC.csv") as file_name:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(file_name)
        # Iterate over each row in the csv using reader object
        k = 0
        for row in csv_reader:
            if k == N_PCs:
                break
            PCs[i,k,:] = np.array(row,dtype=np.float32)
            k+= 1

#%%
N_samples = 100
N_classes = 2
distances = np.zeros((N_samples*N_classes,N_classes))

for i,defect in enumerate(defects[1:N_classes]): #loop over defect type
    print("defect " + defect)
    for j,kernel in enumerate(os.listdir(direc+series+defect)): #loop over images in defect
        if j == N_samples:
            break
        f_name = direc + series+defect+kernel
        Gamma = mpimg.imread(f_name)[:,:,0]
        plt.imshow(Gamma,cmap = "gray")
        plt.title("Image")
        plt.show()
        proj = np.zeros(2500)
        for k in range(N_classes): #test image for each defect type
            for l in range(N_PCs): #loop over Principal components
                PC = PCs[k,l,:]
                if l == 0:
                    avg = PC
                    Omega = Gamma.flatten() - PC

                else:
                    omega= np.dot(Omega,PC)
                    proj += omega*PC
                    
            plt.imshow((proj+avg).reshape((50,50)),cmap = "gray")
            plt.title(["classs" + str(k)])
            plt.show()
            
            distance = np.linalg.norm(proj-Omega)
            print("distance to class "+str(k) + " is "+ str(distance))
            distances[i*N_samples+j,k] = distance
        break
    break

            
#%%

Crack_x = 0
Crack_y = 1
plt.plot(distances[:N_samples,0],distances[:N_samples,1],'*',label = "Crack A")
plt.plot(distances[N_samples:,0],distances[N_samples:,1],'*', label = "Crack B")    
plt.legend()