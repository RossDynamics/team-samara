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

os.chdir('Norway Trial Data')
file_name = 'n-g01-t01-data'

drop = pd.read_csv(file_name+'.csv')
#plt.plot(drop['Row'], drop['Column'])



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

### Pixels per inch in x-direction
#pixels_to_inch = r[2]/.5
#Pixels per inch in y-direction
pixels_to_inch = r[3]/6
#pixels_to_inch = 22.5
frame_sect_beg = 0
W = 300
dt = 1/2000
N = 10000
plt.figure(1)
plt.plot(drop['Column']/pixels_to_inch, drop['Row']/pixels_to_inch)
plt.gca().invert_yaxis()
plt.axis('equal')
frame_end = np.size(drop['Column'])
freq = np.zeros(frame_end-W)
avg_vel_m_s = np.zeros(N-W)

x = drop['Column']/pixels_to_inch
y = drop['Row']/pixels_to_inch
x = x-np.mean(x)
t = drop['FrameNo']
dt_new = t.values[-1]*dt/N
spl = interp.UnivariateSpline(t, x, k = 1, s=0)
ts = np.linspace(np.min(t), np.max(t), N)
yinterp = np.interp(ts, t, y)
interped = spl(ts)
b, a = spsig.butter(3, 0.003)
xs = spsig.filtfilt(b, a, interped)
d, c = spsig.butter(3, 0.003)
ys = spsig.filtfilt(d, c, yinterp)

plt.figure(2)
plt.plot(xs, ys)
plt.gca().invert_yaxis()
plt.axis('equal')

while frame_sect_beg+W < N: 

    frame_sect_end = frame_sect_beg+W
    frame_mid = (frame_sect_beg+frame_sect_end)/2




#    omega = np.linspace(0, 1/(2*dt), N//2)
#    xf = fftpack.fft(xs)
#    xwf = fftpack.fft(xs*spsig.blackman(N))
#    mag = 2/N*np.abs(xf[0:N//2])
#    magw = 2/N*np.abs(xwf[0:N//2])
#
#    test = np.zeros(xwf.shape)
#    ind = np.argmax(np.abs(xwf[3:100]))
#    test[ind+3] = np.abs(xwf[ind+3])
#    testsig = fftpack.ifft(test)
    
#    freq[frame_sect_beg] = np.max(test) #in units?
    avg_vel_in_s = (ys[frame_sect_end]-ys[frame_sect_beg])/(W*dt_new) # in inches per second
    avg_vel_m_s[frame_sect_beg] = avg_vel_in_s/39.37 #in meters per second


    frame_sect_beg = frame_sect_beg+1

plt.figure(3)
plt.plot(avg_vel_m_s)
plt.title('Average velocity of samara')
plt.ylabel('v, m/s')
#plt.figure(2)
#plt.plot(freq)
#plt.title('Frequency of Autorotation')

#plt.figure(3)
#plt.plot(omega, mag)
#plt.plot(omega, magw)
#plt.xlim([0,10])
#
#plt.figure(4)
#plt.plot(ts, xs)
#plt.plot(ts, 10*testsig)
#plt.show()