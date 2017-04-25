
# coding: utf-8

# In[23]:

import cv2
import numpy as np
import matplotlib.pyplot as ploty

im1 =  cv2.imread("u2cuba.jpg",0)
im2 =  cv2.imread("trailer.png",0)

paddedimage = np.zeros((im1.shape[0]+im2.shape[0]+1,im1.shape[1]+im2.shape[1]+1))


# In[2]:

def cross_correlation(img,template,res):
    for i in range(0,img.shape[0]-template.shape[0]+1):
        for j in range(0,img.shape[1]-template.shape[1]+1):
            im = np.array(img[i:i+template.shape[0],j:j+template.shape[1]],copy=True) 
            if im.std() == 0:
                continue
            im = (im - im.mean())/im.std()
            final = np.sum(np.multiply(im,template))
            res[i][j] = final


# In[24]:

cross_correlation(im1,im2,paddedimage)
# print paddedimage


# In[25]:

paddedimage = np.array(paddedimage)
index1,index2 = np.where(paddedimage == np.max(paddedimage))
print(index1,index2)
for x,y in zip(index1,index2):
    print x,y


# In[20]:




# In[26]:

im22 =  cv2.imread("trailerSlightlyBigger.png",0)
paddedimage1 = np.zeros((im1.shape[0]+im22.shape[0]+1,im1.shape[1]+im22.shape[1]+1))

cross_correlation(im1,im22,paddedimage1)
paddedimage1 = np.array(paddedimage)
index11,index22 = np.where(paddedimage1 == np.max(paddedimage1))
print(index11,index22)
for x,y in zip(index11,index22):
    print x,y


# In[24]:

def convolution(imgg,template):
    img = np.lib.pad(imgg,(template.shape[0]-1)/2, 'constant', constant_values=(0))
    resultant = np.zeros(imgg.shape)
    template = np.flipud(np.fliplr(template))
    for i in range(0,img.shape[0]-template.shape[0]+1):
        for j in range(0,img.shape[1]-template.shape[1]+1):
            im = np.array(img[i:i+template.shape[0],j:j+template.shape[1]],copy=True) 
            final = np.sum(np.multiply(im,template))
            resultant[i][j] = final
    return resultant


# In[ ]:
box = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).reshape((3,3))
test = convolution(im1,box)
box = box/9
ploty.imshow(np.array(test),cmap="gray")
# ploty.show()


Gy = np.array([-1.0,-2.0,-1.0,0.0,0.0,0.0,1.0,2.0,1.0]).reshape((3,3))
Gx = np.transpose(Gy)
Gy = Gy/8
Gx = Gx/8
mod = np.sqrt((np.absolute(Gx),np.absolute(Gx))+ (np.absolute(Gy),np.absolute(Gy))) 
lap = np.array([-1.0,-1.0,-1.0,-1.0,8.0,-1.0,-1.0,-1.0,-1.0]).reshape((3,3))
lap = lap/16

img1 = cv2.imread("clown.tif",0)
edge1 = convolution(img1,Gy)

ploty.subplot('242').imshow(np.array(edge1),cmap="gray")

edge2 =  convolution(img1,Gx)

ploty.subplot('243').imshow(np.array(edge2),cmap="gray")



tedge =np.add(np.absolute(edge1),np.absolute(edge2))
ploty.subplot('241').imshow(np.array(tedge),cmap="gray")


tedge1 =  convolution(img1,lap)
ploty.subplot('244').imshow(np.array(tedge1),cmap="gray")


from scipy import signal 

grad = signal.convolve2d(im1, Gx, boundary='fill')
grad1 = signal.convolve2d(im1, Gy, boundary='fill')
grad2 = signal.convolve2d(im1, tedge, boundary='fill') 
grad3 = signal.convolve2d(im1, lap, boundary='fill') 

ploty.subplot('247').imshow(grad,cmap="gray")
ploty.subplot('246').imshow(grad1,cmap="gray") 
ploty.subplot('245').imshow(grad2,cmap="gray")
ploty.subplot('248').imshow(grad3,cmap="gray")
ploty.show()


# In[6]:

import cv2
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as ploty
from scipy.ndimage.filters import gaussian_filter 

im1 =  cv2.imread("cat.bmp")
im2 =  cv2.imread("dog.bmp")

blur1 = gaussian_filter(im2,10)
blur2 = im1 - gaussian_filter(im1,5)

res = blur1 + blur2

ploty.imshow(res,cmap = "gray")
ploty.savefig("one")



# In[6]:

# In[ ]:



