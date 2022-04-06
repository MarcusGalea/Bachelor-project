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


test = False

N_PCs = 400
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
            
    
if test:
    series +=r"test\\"

#%% Classifier type 1
N_samples = 100
N_classes = 4
N_PCs = 100
distances = np.zeros((N_classes,N_samples,N_classes))
PC1_direction = np.zeros((N_classes,N_samples,N_classes))
PC2_direction = np.zeros((N_classes,N_samples,N_classes))

for i,defect in enumerate(defects[0:N_classes]): #loop over defect type
    #print("defect " + defect)
    for j,kernel in enumerate(os.listdir(direc+series+defect)): #loop over images in defect
        if j == N_samples:
            break
        f_name = direc + series+defect+kernel
        Gamma = mpimg.imread(f_name)[:,:,0]
        
        """
        plt.imshow(Gamma,cmap = "gray")
        plt.title("Image")
        plt.show()
        """
        
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
                    if l == 1:
                        PC1_direction[i,j,k] = omega
                    if l == 2:
                        PC2_direction[i,j,k] = omega
                """    
            plt.imshow((proj+avg).reshape((50,50)),cmap = "gray")
            plt.title(["classs" + str(k)])
            plt.show()
            """
            
            distance = np.linalg.norm(proj-Omega)
            #print("distance to class "+str(k) + " is "+ str(distance))
            distances[i,j,k] = distance
            
#%% plot of results



def plot_points(Crack_1 = 2, Crack_2 = 3, N_samples = 20, method = distances):

    plt.plot(method[Crack_1,:,Crack_1],method[Crack_1,:,Crack_2],'*',label = defects[Crack_1])
    plt.plot(method[Crack_2,:,Crack_1],method[Crack_2,:,Crack_2],'*',label = defects[Crack_2])
    plt.xlabel(defects[Crack_1])
    plt.ylabel(defects[Crack_2])
    plt.legend()
    
Crack_1 = 0
Crack_2 = 3
plot_points(Crack_1,Crack_2,N_samples,distances)


#%%

Crack_1 = 1
Crack_2 = 3
plot_points(Crack_1,Crack_2,N_samples,PC1_direction)

#%%

Crack_1 = 0
Crack_2 = 3

plt.plot(PC1_direction[Crack_1,:,Crack_1],PC2_direction[Crack_1,:,Crack_1],'*',label = defects[Crack_1])
plt.plot(PC1_direction[Crack_2,:,Crack_1],PC2_direction[Crack_2,:,Crack_1],'*',label = defects[Crack_2])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()