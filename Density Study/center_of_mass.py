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
area = matrix = np.zeros((30,1))
center_mass = matrix = np.zeros((30,1))
center_percent = matrix = np.zeros((30,1))
total_data = matrix = np.zeros((30,200))
total_data2 = matrix = np.zeros((30,200))
total_data3 = matrix = np.zeros((30,200))
total_data4 = matrix = np.zeros((30,200))
area2 = matrix = np.zeros((30,1))
colors = [mpl.cm.viridis(a) for a in np.linspace(0, 1, 30)]


#Plot mass fraction vs length fraction
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
#    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass fraction, $m$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.ylim([0, 1])
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)), color=colors[a])
for a, c in enumerate(center_percent):
        plt.plot([c, c], [0, 1], color=colors[a])
  
plt.plot([np.mean(center_percent),np.mean(center_percent)],[0,1], color='r')

#Plot mass vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+2][1:21]
    length[a, 1:] = data_list[a+34][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    g = interp1d(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), (Y[~X.mask]), kind='cubic')
    total_data2[a] = g(np.linspace(0,.99,200))
    plt.figure(2)
    plt.plot(np.linspace(0, 0.99, 200), g(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass, $mg$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)


# Find and plot average center of mass percent
for a, c in enumerate(center_percent):
        plt.plot([c, c], [0, 60], color=colors[a])
#Plot Density vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+2][1:21]
    length[a, 1:] = data_list[a+34][1:21]
    area[a] = data_list[a+66][1]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
#    Z = ma.masked_invalid(area[a])
    h = interp1d(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), (Y[~X.mask])/(area[a]), kind='cubic')
    total_data3[a] = h(np.linspace(0,.99,200))
    plt.figure(3)
    plt.plot(np.linspace(0, 0.99, 200), h(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Area Density, $mg/mm^2$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)



#Plot linear density vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+2][1:21]
    length[a, 1:] = data_list[a+34][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    j = interp1d(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), ((Y[~X.mask])/(X[~X.mask]))/(np.sum((Y[~X.mask])/(X[~X.mask]))), kind='cubic')
    total_data4[a] = j(np.linspace(0,.99,200))
    plt.figure(4)
    plt.plot(np.linspace(0, 0.99, 200), j(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Linear Density, $mg/mm$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    area2[a] = np.trapz(total_data4[a],dx=.1)




        
        

        
# Find best fit line for mass fraction vs length fraction
#area = np.trapz(total_data[5],dx=.1)
xdata = np.linspace(0,.99,200)
mean_data = [np.mean(total_data[:, k]) for k in range(len(xdata))]
#xdata = np.vstack([xdata[1:199:1] for k in range(30)]).ravel()
xdata = np.vstack([xdata for k in range(30)]).ravel()
#remove 0 entry from array
#total_data = total_data[:,1:199:1].ravel()
total_data = total_data.ravel()

#fit data
z = np.polyfit(xdata,total_data,9)

xdata = np.linspace(0,.99,200)
y = np.poly1d(z)
plt.figure(1)
plt.plot(xdata, y(xdata), color='r')
#plot mean data
#plt.plot(np.linspace(0,.99,200), mean_data, color='k')

def y2(x):
    return 5.75154970e+02*x**9-2.80978700e+03*x**8+5.88451014e+03*x**7-6.92413516e+03*x**6+5.04191080e+03*x**5-2.35416177e+03*x**4+6.99852513e+02*x**3-1.23416369e+02*x**2+1.00921020e+01*x-1.11407137e-03


#Plot fit by itself
plt.figure(5)
plt.plot(xdata,y(xdata), color='r')
plt.xlabel('Fraction of length', **labelfont)
plt.ylabel('Mass fraction, $m$', **labelfont)

plt.show()
#area = np.trapz(total_data[3],dx=.1)

