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
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import seaborn as sns
sns.set_style('ticks')
mpl.rc('mathtext', rm='serif', fontset='cm')
mpl.rc('font', family='serif', serif='Times New Roman')
labelfont = {'fontsize':9, 'fontname':'Times New Roman'}
tickfont = {'fontsize':8, 'fontname':'Times New Roman'}

#def func(xdata,s,m):
#    return np.array([(1/(s*x*np.sqrt(2*np.pi)))*np.exp((-1/(2*(s**2)))*((np.log(x)-m)**2)) for x in xdata],dtype=np.float64)
 
#def func2(xdata,a,b,c, d):
#    return np.array([a*x**(c-1)*(1-d*np.exp(-b*x**c))*np.exp(-b*x**c) for x in xdata], dtype=np.float64)

def func3(xdata,a,b,c):
    return np.array([a*np.exp(b*x)*np.exp(-(b*x**2))*x+c*np.exp(-x)*x**2 for x in xdata], dtype=np.float64)

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
    f = interp1d(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), (Y[~X.mask])/np.sum(Y[~X.mask]), kind='cubic')
    total_data[a] = f(np.linspace(0,.99,200))
    plt.figure(1)
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass fraction, $m$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.ylim([0, 1.1])
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)), color=colors[a])

# Find and plot average center of mass percent
for a, c in enumerate(center_percent):
    plt.plot([c, c], [0, 1], color=colors[a])
  
plt.plot([np.mean(center_percent),np.mean(center_percent)],[0,1], color='r')



# Find best fit line for mass fraction vs length fraction

xdata = np.linspace(0,.99,200)
mean_data = [np.mean(total_data[:, k]) for k in range(len(xdata))]
#xdata = np.vstack([xdata[1:199:1] for k in range(30)]).ravel()
xdata = np.vstack([xdata for k in range(30)]).ravel()
#remove 0 entry from array
#total_data = total_data[:,1:199:1].ravel()
total_data = total_data.ravel()

#generate intial guess

#fit data
z = np.polyfit(xdata,total_data,9)
#popt, pcov = curve_fit(func3, xdata, total_data)
#shape, loc, scale = lognorm.fit(total_data, floc=0)
#k, loc, scale = stats.exponnorm.fit(total_data, floc=0)
#a, c, loc, scale = stats.exponweib.fit(total_data, floc=0)
#plt.plot(xdata,lognorm.pdf(xdata,shape,scale=scale) , color='r')
#plt.plot(xdata,stats.exponnorm.pdf(xdata,k,scale=scale) , color='r')
#Plot Best fit
#y_fit = func3(xdata,*popt)
y = np.poly1d(z)
plt.plot(xdata, y(xdata), color='r')
plt.plot(np.linspace(0,.99,200), mean_data, color='k')