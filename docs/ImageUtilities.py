#!/usr/bin/env python
# coding: utf-8

# ##### 0. import libraries
import numpy as np
import cv2
import math

# Image Processing Utilities
def binarize_image(img, block_size=151, c=20):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    return img_bin

def trim(img, pixel_threshold=0):
    r1,c1, r2,c2 = 0,0, (img.shape[0]-1),(img.shape[1]-1)
    th = 255 * pixel_threshold
    
    while(np.sum(img[r1]) <= th):
        r1+=1
    
    while(np.sum(img[r2]) <= th):
        r2-=1
        
    while(np.sum(img[:, c1]) <= th):
        c1+=1
        
    while(np.sum(img[:, c2]) <= th):
        c2-=1
        
    img_roi = img[r1:r2, c1:c2]
    return img_roi

def segment_image(img):
    #Segmentation
    m1 = m2 = -1
    flag = ic = ir = 0
    seg_list = []
    
    r = img.shape[0]
    c = img.shape[1]
    print(r, c)
    img = np.c_[np.zeros((r,2)), img]
    img = np.c_[img, np.zeros((r,2))]
    print(r, c)
    r = img.shape[0]
    c = img.shape[1]
    
    for ic in range(c):
            
        if(m1 > -1 and m2 > -1):
            if(m1 == m2):
                continue
            seg_list.append(img[:, (m1-1):(m2+1)])
            m1 = -1
            m2 = -1

        flag = 0
        for ir in range(r):
            if(img[ir, ic] > 0):
                flag = 1
                break

        if(flag == 1 and m1 == -1):
            m1 = ic
        elif(flag == 0 and m1 > -1):
            m2 = ic
    
    #Height Cropping all the segments
    new_list = []
    for ix in range(len(seg_list)):
        char_seg = seg_list[ix]
        r = char_seg.shape[0]
        c = char_seg.shape[1]
        
        r1 = r2 = -1
        for i in range(r):
            for j in range(c):
                if(char_seg[i, j] == 255):
                    r1 = i
                    break
            if(r1 > -1):
                break        
        for i in range(r-1, 0, -1):
            for j in range(c):
                if(char_seg[i, j] == 255):
                    r2 = i
                    break
            if(r2 > -1):
                break
        if(r1==r2):
            continue
        new_list.append(char_seg[r1:r2, :])
        
    return new_list

def flatten_segments(seg_list):
    #Resize to 28 x 28
    new_seg_list = []
    for i in range(len(seg_list)): 
        img = seg_list[i]
        
        #Flatteing
        img_flat = np.reshape(img, (28*28))
        new_seg_list.append(img_flat)
    
    return new_seg_list

def resize_segments(seg_list):
    #Resize to 28 x 28
    new_seg_list = []

    for i in range(len(seg_list)):
        img_re = cv2.resize(seg_list[i], (28, 28), interpolation=cv2.INTER_AREA)
        new_seg_list.append(img_re)
        
    return new_seg_list

# User Image Processing
def getCenterOfMass(ar2d):
    rsum = 0.0
    csum = 0.0
    total =0.0
    for ir in range(ar2d.shape[0]):
        for ic in range(ar2d.shape[1]):
            rsum += ir * ar2d[ir, ic]
            csum += ic * ar2d[ir, ic]
            total += ar2d[ir, ic]
    
    cx = csum / total
    cy = rsum / total
    
    return cx, cy

def getTransform(img):
    r, c = img.shape
    cx, cy = getCenterOfMass(img)
    shX = int(np.round(c/2 - cx))
    shY =  int(np.round(r/2 - cy))
    
    return shX, shY

def transformImage(img, shX, shY):
    transform = np.array([[1, 0, shX],
                [0, 1, shY]]).astype(np.float32)
    r, c = img.shape
    img_transformed = cv2.warpAffine(img, transform, (c, r))
    
    return img_transformed

def fitImage(img, max_dim):
    rows, cols = img.shape
    img_fit = img
    if rows > cols:
            factor = max_dim/rows
            rows =max_dim
            cols = int(round(cols*factor))
            img_fit = cv2.resize(img, (cols,rows))
    else:
        factor = max_dim/cols
        cols = max_dim
        rows = int(round(rows*factor))
        img_fit = cv2.resize(img, (cols, rows))
    return img_fit

def padImage(img, reqr, reqc):
    rows, cols = img.shape
    
    row_t = int(math.ceil((reqr-rows)/2.0))
    col_l = int(math.ceil((reqc-cols)/2.0))
    
    img_padded = np.zeros((reqr, reqc))
    i=j=0
    for ir in range(row_t, (int)(row_t+rows)):
        j=0
        for ic in range(col_l, (int)(col_l+cols)):
            img_padded[ir, ic] = img[i, j]
            j+=1
        i+=1
    
    return img_padded    

