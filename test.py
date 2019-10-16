# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:46:14 2019

@author: Elena
"""
from PIL import Image


from skimage.io import imread, imshow, imsave
import numpy
from math import floor, ceil
import scipy
from numba import jit

@jit
def median_count (img1):
    [x1,y1,z1]=numpy.shape(img1)
    median_img1=numpy.zeros([y1,z1])
    for z in range(z1):
        for y in range(y1): 
            median_img1[y][z]=numpy.median(img1[:,y,z])
    return (median_img1)   
@jit
def mean_filter (img2):
    mask = numpy.array([[-1/9,-1/9,-1/9], [-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]], dtype=numpy.float64)
    img_out = scipy.signal.convolve2d(img2,mask,mode='same')
    return img_out
def join_img(i1,i2,i3,i4,i5):
    j_img = numpy.stack((i1,i2, i3,i4,i5))
    return j_img

img1 = imread('1 (1).jpg', as_gray=True)
img2 = imread('1 (2).jpg', as_gray=True)
img3 = imread('1 (3).jpg', as_gray=True)
img4 = imread('1 (4).jpg', as_gray=True)
img5 = imread('1 (5).jpg', as_gray=True)
img_1=mean_filter(img1)
img_2=mean_filter(img2)
img_3=mean_filter(img3)
img_4=mean_filter(img4)
img_5=mean_filter(img5)
img_st = join_img(img_1,img_2,img_3,img_4,img_5)

img_med=median_count(img_st)
[n,m]=numpy.shape(img_med)
new_img=numpy.ones([n,m])
m1=0
for i in range(n):
    for j in range(m):
        if img_med[i][j]>0:
            new_img[i][j] =0
            m1=m1+1

imsave('med_I.jpg',255*new_img)
#imsave('F_11.jpg',img_2)

