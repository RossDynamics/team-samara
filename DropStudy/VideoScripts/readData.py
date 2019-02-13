# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:17:51 2019

@author: Norris 111
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.signal as spsig
from scipy import fftpack

# os.chdir('Norway Trial Data')
file_name = 'n-g01-t01-data'

drop = pd.read_csv('Norway Trial Data/'+file_name+'.csv')
pixels_to_inch = 22.5
x = drop['Column']/pixels_to_inch
y = drop['Row']/pixels_to_inch
x = x-np.mean(x)
dt = 1/2000
t = drop['FrameNo']*dt


N = 10000
spl = interp.UnivariateSpline(t, x, k = 1, s=0)
ts = np.linspace(np.min(t), np.max(t), N)
yinterp = np.interp(ts, t, y)
interped = spl(ts)
b, a = spsig.butter(4, 0.008)
xs = spsig.filtfilt(b, a, interped)
d, c = spsig.butter(3, 0.008)
ys = spsig.filtfilt(d, c, yinterp)


omega = np.linspace(0, 1/(2*dt), N//2)
xf = fftpack.fft(xs)
xwf = fftpack.fft(xs*spsig.blackman(N))
mag = 2/N*np.abs(xf[0:N//2])
magw = 2/N*np.abs(xwf[0:N//2])

test = np.zeros(xwf.shape)
test[13] = np.abs(xwf[13])
testsig = fftpack.ifft(test)

plt.figure(1)
plt.plot(omega, mag)
plt.plot(omega, magw)
plt.xlim([0,20])
plt.figure(2)
plt.plot(xs, ys)
plt.gca().invert_yaxis()
# plt.gca().invert_yaxis()
plt.axis('equal')
plt.figure(3)
plt.plot(ts, xs)
plt.plot(ts, 10*testsig)
plt.show()
#os.chdir('..')
#os.chdir('2017July07')
#os.chdir('White Background')
#for file in os.listdir('.'):
#   if file[:3] == 'n-g':
#  if file[:9] == file_name[:9]:
#    base = file
#    os.chdir(base)
#    for file in os.listdir('.'):
#      if file[-4:] == '.avi':
#        vidname = file
        
#video = cv2.VideoCapture(vidname)

#ok, frame = video.read()

##Choose box that covers  inches, and the width of the tape
#r = cv2.selectROI(frame, False, False)

## Pixels per inch in x-direction
#pixels_to_inch = r[3]/6


# os.chdir('..')
#print(pixels_to_inch)
##Pixels per inch in y-direction
#y_conv = r[3]/6
