

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat

k = 0
direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Bachelor-project--defect-detection-on-solar-panels\data\\"
series = r"Series6\\"
images = direc + series + r"CellsCorr_noline\\"
labels = direc + series + r"MaskGT\\"

y = []
dic = {}
k = 0

n = 10
m = 6
for pic in os.listdir(images):
    if pic != "Thumbs.db":
        #txt = pic.split("Corr")[1]
        #til series 6
        txt = pic.split("ImageGS")[1]
        txt = txt.split(".")[0]
        dic[txt] = k
        y.append([pic,[0,0,0,0]])
        k += 1

for label in os.listdir(labels):
    txt = label.split("Image")[1]
    txt = txt.split(".")[0]
    vec = np.zeros(4)
    #vec = [Crack A, Crack B, Crack C, Finger Failure]
    mat_label = loadmat(labels + label)['GTLabel']
    try:
        for i in range(len(mat_label)):
            print(mat_label[i][0][0])
            if mat_label[i][0][0] == 'Finger Failure':
               y[dic[txt]][1][3]  = 1
            if mat_label[i][0][0] == 'Crack A':
                y[dic[txt]][1][0] = 1
            if mat_label[i][0][0] == 'Crack B':
                y[dic[txt]][1][1] = 1
            if mat_label[i][0][0] == 'Crack C':
                y[dic[txt]][1][2] = 1 
    except KeyError:
        print("cells for "+txt+" are missing")
        continue

pd.DataFrame(y).to_csv(direc + series + "labels2.csv",header = None, index = None)

