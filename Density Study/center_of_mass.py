# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:05:57 2018

@author: Nathaniel
"""

import pandas as pd
import math
import numpy as np
from numpy import ma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import seaborn as sns
sns.set_style('ticks')
mpl.rc('mathtext', rm='serif', fontset='cm')
mpl.rc('font', family='serif', serif='Times New Roman')
labelfont = {'fontsize':9, 'fontname':'Times New Roman'}
tickfont = {'fontsize':8, 'fontname':'Times New Roman'}

def func(x,s):
    l = np.array([(1/(s*x*np.sqrt(2*np.pi)))*np.exp((-1/2)*(np.log(x)/s)**2)],dtype=np.float64)
    return l

 
df = pd.read_excel('Samara Segment Mass.xlsx', sheet_name='Sheet1')
data_list = df.values.tolist()

mass_length = matrix = np.zeros((30,21))
mass = matrix = np.zeros((30,21))
length = matrix = np.zeros((30,21))
center_mass = matrix = np.zeros((30,1))
center_percent = matrix = np.zeros((30,1))
total_data = matrix = np.zeros((30,200))

colors = [mpl.cm.viridis(a) for a in np.linspace(0, 1, 30)]


# Iterate over 30 samaras
for a in range(0,30):
    mass[a, 1:] = data_list[a+2][1:21]
    length[a, 1:] = data_list[a+34][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    f = interp1d(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), Y[~X.mask]/np.sum(Y[~X.mask]), kind='cubic') 
    total_data[a] = f(np.linspace(0,.99,200))
    plt.figure(1)
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass fraction, $m$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.ylim([0, 0.6])
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)), color=colors[a])

# Find and plot average center of mass percent
for a, c in enumerate(center_percent):
  plt.plot([c, c], [0, 1], color=colors[a])
  
plt.plot([np.mean(center_percent),np.mean(center_percent)],[0,1], color='r')



# Find best fit line for mass fraction vs length fraction

xdata = np.linspace(0,.99,200)
xdata = xdata[1:199:1]
#remove 0 entry from array
total_data = total_data[:,1:199:1]
#generate intial guess
y = func(xdata,.6)
#fit data
popt, pcov = curve_fit(func, xdata, total_data)