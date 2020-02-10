# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:38 2019

@author: Mohammadmahdi
"""

import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import mylib as ml
import matchlib as mat

img1 = cv.imread("1.jpg",0)
img2 = cv.imread("2.jpg",0)


m,n = np.shape(img1)


h,x = ml.hist(img1)
s,_ = ml.Chist(img1)
img3 = ml.histeq(img1)
hnew,xnew = ml.hist(img3)
snew,_ = ml.Chist(img3)
img4 = ml.histeq(img2)
img5 = mat.imhistmatch(img3,img4)
img5 = img5.astype("uint8")
plt.figure(1)
plt.subplot(221),plt.bar(x,h,width = 0.8,bottom = 0,align = 'center',),plt.show()
plt.title('Histogram function of input image'),plt.ylabel('PDFr(r)')
plt.subplot(222),plt.bar(x,s,color='g',width = 0.8,bottom = 0,align = 'center',),plt.show()
plt.title('CHistogram function of input image'),plt.ylabel('CDFr(r)')
plt.subplot(223),plt.bar(x,hnew,color='r',width = 0.8,bottom = 0,align = 'center',),plt.show()
plt.title('Equalized Histogram function of input image'),plt.ylabel('PDFs(s)')
plt.subplot(224),plt.bar(x,snew,color='c',width = 0.8,bottom = 0,align = 'center',),plt.show()
plt.title('Equalized CDF function of input image'),plt.ylabel('CDFs(s)')

plt.figure(2)
plt.subplot(211)
plot_image1 = np.concatenate((img1, img3), axis=1)
plt.imshow(plot_image1,cmap = "gray"),plt.title('input image1 and histogram equalized image1 respectively')
plt.colorbar()
plt.subplot(212)
plot_image2 = np.concatenate((img2, img4), axis=1)
plt.imshow(plot_image2,cmap = "gray"),plt.title('input image2 and histogram equalized image2 respectively')
plt.colorbar()
plt.show()


plt.figure(3)
plt.subplot(131),plt.imshow(img3,cmap = "gray")
plt.title('Equalized Histogram image1'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(img4,cmap = "gray")
plt.title('Equalized Histogram image2'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img5,cmap = "gray")
plt.title('Matching Histogram result of image one and two'),plt.xticks([]),plt.yticks([])

filename1 = 'Matched image1.tif'
filename2 = 'Target image.tif'
filename3 = 'overlaped images.tif'
cv.imwrite(filename1,img5)
cv.imwrite(filename2,img4)


img6 = np.zeros((m,n,3),dtype = "uint8")
img6[:,:,0] = img5
img6[:,:,1] = img4

cv.imwrite(filename3,img6)


