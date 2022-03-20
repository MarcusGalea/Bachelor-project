# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:50:47 2022

@author: Marcu
"""
import numpy as np

from itertools import groupby
from operator import itemgetter

#%% Image preprocessing

def grassfire(img, N,p):
    # Function which finds pixels that deviate by more than p percent
    # from the average pixelvalue in an NxN neighborhood
    
    #INPUT
    #img: binary or greyscale image
    #N: half of width of neighborhood
    #p: percentwise difference required to denote a pixel as an "island"
    
    #OUTPUT
    #Binary image, showing the "islands" in the image with the pixelvalue "1"
    
    
    n,m = np.shape(img)
    mask = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            l = N
            r = N
            u = N
            d = N
            if i < N:
                u = i
            elif i > n-N:
                d = n-i-1
            if j < N:
                l = j
            elif j > m-N:
                r = m-j-1
            kern = img[i-u:i+d,j-l:j+r]
            nk,mk = np.shape(kern)
            avg = sum(sum(kern))/(nk*mk)
            deviate = abs((avg-img[i,j])/avg)
            if deviate > 0.15:
                mask[i,j] = 1
    return mask


def find_lines(img,tol):
    #finds vertical black lines in an image
    #
    #INPUT
    #img: greyscale image
    #tol: line pixelvalue threshold
    
    #OUTPUT
    #lines: column indexes for vertical lines

    n,m = np.shape(img)
    lines = []
    for j in range(1,m):
        avg = sum(img[:,j])/n
        if avg < tol:
            lines.append(j)
    return lines

def remove_lines(img,lines):
    #removes vertical lines from image
    
    #INPUT
    #img: grayscale image
    #lines: column index for vertical lines
    
    #OUTPUT
    #img: Image without vertical lines
    lines = sorted(set(lines))
    gaps = [[s, e] for s, e in zip(lines, lines[1:]) if s+1 < e]
    edges = iter(lines[:1] + sum(gaps, []) + lines[-1:])
    edges =  list(zip(edges, edges))
    for edge in edges:
        try:
            filt = (img[:,edge[0]-3] + img[:,edge[1]+3])/2
        except IndexError:
            if edge[0]<3:
                filt = img[:,edge[1]+3]
            else:
                filt = img[:,edge[0]-3]
            continue
                
        for line in range(edge[0]-2,edge[1]+2):
            img[:,line] = filt
    return img
        
        

        
        
