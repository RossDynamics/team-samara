# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:05:57 2018

@author: Nathaniel
"""

import pandas as pd
import numpy as np
from numpy import ma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('ticks')
mpl.rc('mathtext', rm='serif', fontset='cm')
mpl.rc('font', family='serif', serif='Times New Roman')
labelfont = {'fontsize':9, 'fontname':'Times New Roman'}
tickfont = {'fontsize':8, 'fontname':'Times New Roman'}

 
df = pd.read_excel('Samara Segment Mass.xlsx', sheet_name='Sheet1')
data_list = df.values.tolist()

mass_length = matrix = np.zeros((30,21))
mass = matrix = np.zeros((30,21))
length = matrix = np.zeros((30,21))
center_mass = matrix = np.zeros((30,1))
center_percent = matrix = np.zeros((30,1))

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
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass fraction, $m$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.ylim([0, 0.6])
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))