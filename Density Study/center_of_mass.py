# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:05:57 2018

@author: Nathaniel
"""

import pandas as pd
import numpy as np

 
df = pd.read_excel('Samara Segment Mass.xlsx', sheet_name='Sheet1')
data_list = df.values.tolist()

mass_length = matrix = np.zeros((30,20))
mass = matrix = np.zeros((30,20))
length = matrix = np.zeros((30,20))
center_mass = matrix = np.zeros((30,1))
center_percent = matrix = np.zeros((30,1))

for a in range(0,30):
    mass[a] = data_list[a+2][1:21]
    length[a] = data_list[a+34][1:21]
    mass_length[a] = [b*c for b,c in zip(mass[a],np.cumsum(length[a]))]
    center_mass[a] = np.nansum(mass_length[a])/np.nansum(mass[a])
    center_percent[a] = center_mass[a]/np.nansum(length[a])

