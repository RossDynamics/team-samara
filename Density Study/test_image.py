# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:46:43 2018

@author: Nathaniel
"""

from __future__ import print_function
from scipy.spatial import distance as dist
import numpy as np
import cv2
from imutils import perspective
from imutils import contours
import argparse
import imutils
import matplotlib.pyplot as plt
import skimage.morphology as skimo
import numpy.ma as ma
from PIL import Image



def norm(image):

  # for m in range(len(image[0, :])):

  #   for n in range(len(image[:, 0])):

  #     if image[n, m] < 0.03

  #       image[n, m] = 0

  # image = image-np.min(image)

  binary = image > 0.2

  dilated = skimo.binary_dilation(binary, skimo.disk(2))

  cleaned = skimo.remove_small_objects(skimo.remove_small_holes(dilated), min_size=800000)

  image = ma.array(image-np.min(image), mask=~cleaned).filled(0)

  if np.max(image) > 0:

    image = image/np.max(image)

  return image


img = Image.new('RGB', (4000, 6000), (139, 770, 40))
# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread("DSC_0061.jpg")
sub = image-img
sub = sub[:,:,2]-sub[:,:,0]
#image = image[:,2000:,:]
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sub = cv2.line(sub,(0,0),(4000,0),(0,0,0),200)
sub = cv2.line(sub,(0,6000),(4000,6000),(0,0,0),200)
#image2 = image[:, :, 2]-image[:,:,0]
#gray = cv2.erode(image,None,iterations=10)
gray = sub/256
gray1 = norm(gray)
gray2 = cv2.GaussianBlur(gray1, (27, 27), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
#imageio.imwrite('temp_gray.jpg', gray2)
#gray3 = cv2.imread('temp_gray.jpg',0)
gray3 = np.uint8(gray2*256)
edged = cv2.Canny(gray3, 0, 20,apertureSize = 3)
edged = cv2.dilate(edged, None, iterations=8)
edged = cv2.erode(edged, None, iterations=4)


plt.figure(figsize=(10, 10))
plt.imshow(gray1)