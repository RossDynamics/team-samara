import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

real_vel = pd.read_csv('MeanVelocityReal.csv',names=['trial','velocity'])
norway_vel = pd.read_csv('MeanVelocityNorway.csv',names=['trial','velocity'])


vals =list(real_vel['velocity'].values)
Rvelocities  = []; 
for val in vals:
    if not ' ' in val:
        Rvelocities.append(float(val))

print('Real Norway maples \nMean: ',np.mean(Rvelocities),'\n','Std: ',np.std(Rvelocities),'\n\n')

plt.figure()
plt.hist(Rvelocities)

vals = list(norway_vel['velocity'].values)
Nvelocities = []; 
for val in vals:
    if not ' ' in val:
        Nvelocities.append(float(val))

print('3D-printed Norway maples \nMean: ',np.mean(Nvelocities),'\n','Std: ',np.std(Nvelocities),'\n\n')
plt.figure()
plt.hist(Nvelocities)
