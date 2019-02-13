# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:17:51 2019

@author: Norris 111
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

os.chdir('Norway Trial Data')
file_name = 'n-g01-t01-data'

open(file_name+'.csv', 'r')


os.chdir('..')
os.chdir('2017July07')
#os.chdir('White Background')
for file in os.listdir('.'):
#   if file[:3] == 'n-g':
  if file[:9] == file_name[:9]:
    base = file
    os.chdir(base)
    for file in os.listdir('.'):
      if file[-4:] == '.avi':
        vidname = file
        
video = cv2.VideoCapture(vidname)

ok, frame = video.read()

##Choose box that covers  inches, and the width of the tape
r = cv2.selectROI(frame, False, False)

## Pixels per inch in x-direction
pixels_to_inch = r[2]/.5 
##Pixels per inch in y-direction
#y_conv = r[3]/6