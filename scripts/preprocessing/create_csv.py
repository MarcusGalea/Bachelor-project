# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:49:47 2022

@author: Marcu
"""

import os
import pandas as pd

k = 0
direc = r'C:\Users\aleks\OneDrive\Skole\DTU\6. Semester\Bachelor Projekt\data\\'
#direc = r"C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\6. Semester\Bachelorprojekt\Bachelor-project--defect-detection-on-solar-panels\data\\"
series = r"Full_data\\"
images = direc + series + r"All_images\\"
labels = direc + series + r"All_masks\\"

y = []
dic = {}
k = 0

n = 10
m = 6
for pic in os.listdir(images):
    if pic != "Thumbs.db":
        txt = pic.split("Corr")[1]
        txt = txt.split(".")[0]
        dic[txt] = k
        y.append([pic,0])
        k += 1

for label in os.listdir(labels):
    txt = label.split("Image")[1]
    txt = txt.split(".")[0]
    try:
        y[dic[txt]][1] = 1
    except KeyError:
        print("cells for "+txt+" are missing")
        continue

pd.DataFrame(y).to_csv(direc + series + "all_labels.csv",header = None, index = None)

