# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:05:57 2018

@author: Nathaniel
"""

import pandas as pd
import numpy as np
from numpy import ma
import scipy.interpolate as interp
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

#Initialize variables and arrays
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
total_data5 = matrix = np.zeros((2,200))
area2 = matrix = np.zeros((30,1))
colors = [mpl.cm.viridis(a) for a in np.linspace(0, 1, 30)]
mass2 = matrix = np.zeros((2,21))
length2 = matrix = np.zeros((2,21))
mass_length2 = matrix = np.zeros((2,21))
center_mass2 = matrix = np.zeros((2,1))
center_percent2 = matrix = np.zeros((2,1))
center_percent2 = matrix = np.zeros((2,1))

#Plot mass fraction vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+1][1:21]
    length[a, 1:] = data_list[a+33][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    Xperc = np.cumsum(X[~X.mask])/np.sum(X[~X.mask])
    Yperc = Y[~X.mask]/np.sum(Y[~X.mask])
    f = interp.UnivariateSpline(Xperc, Yperc, k=3, s=0)
    total_data[a] = f(np.linspace(0,.99,200))
    plt.figure(1)
    plt.title("Mass Fraction vs Length Fraction for Natural Samaras (fit in red)")
#    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)))
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass fraction, $m$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.ylim([0, 1])
    plt.plot(np.linspace(0, 0.99, 200), f(np.linspace(0, 0.99, 200)), color=colors[a])

for a, c in enumerate(center_percent):
    plt.figure(2)
    plt.title("Center of Mass at Length Fraction for Natural Samaras (average in red)")
    plt.xlabel("Fraction of Length")
    plt.ylabel("Center of Mass")
    plt.xlim([0,1])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    plt.plot([c, c], [0, 1], color=colors[a])
    plt.plot([np.mean(center_percent),np.mean(center_percent)],[0,1], color='r')
  


#Plot mass vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+1][1:21]
    length[a, 1:] = data_list[a+33][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    g = interp.UnivariateSpline(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), (Y[~X.mask]), k=3, s=0)
    total_data2[a] = g(np.linspace(0,.99,200))
    plt.figure(3)
    plt.title("Mass vs Length Fraction for Natural Samaras")
    plt.plot(np.linspace(0, 0.99, 200), g(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Mass, $mg$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)


#Plot Density vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+1][1:21]
    length[a, 1:] = data_list[a+33][1:21]
    area[a] = data_list[a+65][1]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
#    Z = ma.masked_invalid(area[a])
    h = interp.UnivariateSpline(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), (Y[~X.mask])/(area[a]), k=3, s=0)
    total_data3[a] = h(np.linspace(0,.99,200))
    plt.figure(4)
    plt.title("Area Density vs Length Fraction for Natural Samaras")
    plt.plot(np.linspace(0, 0.99, 200), h(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Area Density, $mg/mm^2$', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)



#Plot linear density vs length fraction
for a in range(0,30):
    mass[a, 1:] = data_list[a+1][1:21]
    length[a, 1:] = data_list[a+33][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])
    X = ma.masked_invalid(length[a])
    Y = ma.masked_invalid(mass[a])
    j = interp.UnivariateSpline(np.cumsum(X[~X.mask])/np.sum(X[~X.mask]), ((Y[~X.mask])/(X[~X.mask]))*np.sum(X[~X.mask])/np.sum(Y[~X.mask]), k=3, s=0)
    total_data4[a] = j(np.linspace(0,.99,200))
    plt.figure(5)
    plt.title("Linear Density vs Length Fraction for Natural Samaras (fit in red)")
    plt.plot(np.linspace(0, 0.99, 200), j(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Linear Density', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    area2[a] = np.sum((((Y[~X.mask])/(X[~X.mask]))*np.sum(X[~X.mask])/np.sum(Y[~X.mask]))*((X[~X.mask])/np.sum(X[~X.mask])))



for a in range(0,2):
    mass2[a, 1:] = data_list[a+97][1:21]
    length2[a, 1:] = data_list[a+100][1:21]
    mass_length2[a] = [b*c for b,c in zip(mass2[a],np.cumsum(length2[a]))]
    center_mass2[a] = np.nansum(mass_length2[a])/np.nansum(mass2[a])
    center_percent2[a] = center_mass2[a]/np.nansum(length2[a])
    X2 = ma.masked_invalid(length2[a])
    Y2 = ma.masked_invalid(mass2[a])
    w = interp.UnivariateSpline(np.cumsum(X2[~X2.mask])/np.sum(X2[~X2.mask]), ((Y2[~X2.mask])/(X2[~X2.mask]))*np.sum(X2[~X2.mask])/np.sum(Y2[~X2.mask]), k=3, s=0)
    total_data5[a] = w(np.linspace(0,.99,200))
    plt.figure(6)
    plt.title("Linear Density vs Length Fraction for 3D Printed PLA Samaras (natural fit in red)")
    plt.plot(np.linspace(0, 0.99, 200), w(np.linspace(0, 0.99, 200)), color=colors[a])
    plt.xlabel('Fraction of length', **labelfont)
    plt.ylabel('Linear Density', **labelfont)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], **tickfont)
    area2[a] = np.sum((((Y[~X.mask])/(X[~X.mask]))*np.sum(X[~X.mask])/np.sum(Y[~X.mask]))*((X[~X.mask])/np.sum(X[~X.mask])))
        
        

        
# Find best fit line for mass fraction vs length fraction
#area = np.trapz(total_data[5],dx=.1)
xdata = np.linspace(0,.99,200)
mean_data = [np.mean(total_data[:, k]) for k in range(len(xdata))]
mean_data2 = [np.mean(total_data4[:, k]) for k in range(len(xdata))]
std_data = [np.std(total_data[:, k]) for k in range(len(xdata))]
std_data2 = [np.std(total_data4[:, k]) for k in range(len(xdata))]
#xdata = np.vstack([xdata[1:199:1] for k in range(30)]).ravel()
xdata = np.vstack([xdata for k in range(30)]).ravel()
#remove 0 entry from array
#total_data = total_data[:,1:199:1].ravel()
total_data = total_data.ravel()
total_data4 = total_data4.ravel()

#fit data
z = np.polyfit(xdata,total_data,9)
z2 = np.polyfit(xdata,total_data4,9)

xdata = np.linspace(0,.99,200)
y = np.poly1d(z)
y2 = np.poly1d(z2)
plt.figure(1)
plt.plot(xdata, y(xdata), color='r')
plt.figure(5)
plt.plot(xdata, y2(xdata),color='r')
plt.figure(6)
plt.plot(xdata, y2(xdata),color='r')
#plot mean data
#plt.plot(np.linspace(0,.99,200), mean_data, color='k')



#Plot fit by itself
#plt.figure(5)
#plt.plot(xdata,y(xdata), color='r')
#plt.xlabel('Fraction of length', **labelfont)
#plt.ylabel('Mass fraction, $m$', **labelfont)

plt.show()
#area = np.trapz(total_data[3],dx=.1)

plt.figure(7)
ax = plt.gca()
ax.errorbar(xdata, mean_data, yerr=std_data)
ax.plot(xdata,y(xdata), color='r')
plt.title("Standard Deviation for Mass Fraction Fit")

plt.figure(8)
ax = plt.gca()
ax.errorbar(xdata, mean_data2, yerr=std_data2)
ax.plot(xdata,y2(xdata), color='r')
plt.title("Standard Deviation for Linear Density Fit")


