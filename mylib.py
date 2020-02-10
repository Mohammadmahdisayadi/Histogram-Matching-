# -*- coding: utf-8 -*-
# this library is written by mohammad mahdi sayadi
# mohammadmahdisayadi9323@gmail.com

import numpy as np

def hist(img):
    m,n = np.shape(img)
    m = m*n
    H = np.reshape(img,(1,m))
    h = np.zeros((256,))
    x = np.transpose(np.linspace(0,255,256))
    for i in range(256):
        count = 0
        for j in range(m):
            if H[0,j] == i:
                count = count + 1
        h[i] = count
    
    
    return h,x


def Chist(img):
    M,N = np.shape(img)
    h,_ = hist(img)
    cdf = np.cumsum(h)
    cdf_normalized = cdf*np.max(h)/ np.max(cdf)
    return cdf_normalized,cdf


def histeq(img):
    _,cdf = Chist(img)
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - np.min(cdf_m))*255/(np.max(cdf_m)-np.min(cdf_m))
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    return img2


def imadjust(ref,img):
    M,N = np.shape(img)
    _,G = Chist(ref)
    G = np.round(G)
    s,_ = Chist(img)
    
    
        