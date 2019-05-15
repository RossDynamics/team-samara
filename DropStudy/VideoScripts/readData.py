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

def datarun(filename):
    file_name = filename
    drop = pd.read_csv(file_name)
    #plt.plot(drop['Row'], drop['Column'])
#    os.chdir('..')
#    os.chdir('2017July07')
#    #os.chdir('White Background')
#    for file in os.listdir('.'):
#    #   if file[:3] == 'n-g':
#      if file[:9] == file_name[:9]:
#        base = file
#        os.chdir(base)
#        for file in os.listdir('.'):
#          if file[-4:] == '.avi':
#            vidname = file
#            
#    video = cv2.VideoCapture(vidname)
#    
#    ok, frame = video.read()
    
    ##Choose box that covers  inches, and the width of the tape
#    r = cv2.selectROI(frame, False, False)
    
    ### Pixels per inch in x-direction
    #pixels_to_inch = r[2]/.5
    #Pixels per inch in y-direction
#    pixels_to_inch = r[3]/6
    #pixels_to_inch = 22.5
    frame_sect_beg = 0
    W = 1500
    if filename[8] == '3':
        dt = 1/3000
        pixels_to_inch = 34
    else:
        dt = 1/2000
        pixels_to_inch = 23.165
            
    N = 10000
#    plt.figure(1)
#    plt.plot(drop['Column']/pixels_to_inch, drop['Row']/pixels_to_inch)
#    plt.gca().invert_yaxis()
#    plt.axis('equal')

    avg_vel_m_s = np.zeros(N-W)

    x = drop['Column']/pixels_to_inch
    y = drop['Row']/pixels_to_inch
    t = drop['FrameNo']
    x = x-np.mean(x)
    
    def removeJumps(X, Y):
     ds = np.sqrt(np.diff(X)**2+np.diff(Y)**2)
     jumps = ds < 3*np.mean(ds)
     if np.sum(jumps) == len(ds):
       return True, X, Y
     else:
       indexlist = np.where(jumps==True)
       start = indexlist[0][0]
       end = indexlist[0][-1]
       x = X[start:end+1]; y = Y[start:end+1]
       jumps = jumps[start:end+1]
       t = np.linspace(0, 1, len(x))
       splx = interp.interp1d(t[jumps], x[jumps])
       sply = interp.interp1d(t[jumps], y[jumps])
       return False, splx(t), sply(t)
    
    good = False
    
    while not good:
        good, x, y = removeJumps(x,y)
    
#    plt.plot(x,y)
    t = t[:len(x)]
    dt_new = t.values[-1]*dt/N
    spl = interp.UnivariateSpline(t, x, k = 1, s=0)
    ts = np.linspace(np.min(t), np.max(t), N)
    yinterp = np.interp(ts, t, y)
    interped = spl(ts)
    b, a = spsig.butter(3, 0.003)
    xs = spsig.filtfilt(b, a, interped)
    d, c = spsig.butter(3, 0.003)
    ys = spsig.filtfilt(d, c, yinterp)
    
#    plt.figure(2)
#    plt.plot(xs, ys)
#    plt.gca().invert_yaxis()
#    plt.axis('equal')
    
    while frame_sect_beg+W < N: 
        frame_sect_end = frame_sect_beg+W
        avg_vel_in_s = (ys[frame_sect_end]-ys[frame_sect_beg])/(W*dt_new) # in inches per second
        avg_vel_m_s[frame_sect_beg] = avg_vel_in_s/39.37 #in meters per second
        frame_sect_beg = frame_sect_beg+1
    
    x_vals = range(0,np.size(avg_vel_m_s))
#    Z = np.poly1d(np.polyfit(x_vals, avg_vel_m_s, 5))
    def findCutoff(T, v):
       for cutoff in range(len(T[:-1000])):
           ave = np.mean(v[cutoff:])
    #       std = np.std(v[cutoff:])
           if v[cutoff]-ave < .1:
               return cutoff
       return False
    
    cutoff = findCutoff(ts,avg_vel_m_s)
#    print(cutoff)
    AVG = np.mean(avg_vel_m_s[cutoff:])
    peaks1, _ = spsig.find_peaks(xs[cutoff:])
    peaks2, _ = spsig.find_peaks(xs[cutoff:], distance=np.mean(np.diff(peaks1))/4)
    time_peaks = [ts[cutoff+pk]*dt for pk in peaks2]
    period = np.mean(np.diff(time_peaks))
    frequency = 2*np.pi/period
#    print(AVG)
#    plt.figure(3)
#    plt.plot(avg_vel_m_s)
#    plt.plot([cutoff,cutoff],[np.min(avg_vel_m_s),np.max(avg_vel_m_s)])
    #plt.plot(xs[:np.size(avg_vel_m_s)], func(xs[:np.size(avg_vel_m_s)], *popt), 'r-', label="Fitted Curve")
#    plt.plot(x_vals,Z(x_vals),'r-')
#    plt.title('Average velocity of samara')
#    plt.ylabel('v, m/s')
    
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

    return cutoff, AVG, frequency
    
    
  #main loop

folders = ['Silver Trial Data', 'Real Samara Data', 'Norway Trial Data']
files = ['SilverMapleData','RealSamaraData','NorwayMapleData']

for fname, folder in zip(files, folders):
    os.chdir(folder)
    with open('../'+fname+'.csv', 'w') as data:
        data.write('Trial,Cutoff,Avg Vel,Ang Vel\n')
        for file in os.listdir('.'):
            c, v, f = datarun(file)
            if v > 2.2 or v < 0.3:
                v = ' '
                c = ' '
            if np.isnan(f):
                f = ' '
            data.write('{},{},{},{}\n'.format(file, c,v,f))
    os.chdir('..')
