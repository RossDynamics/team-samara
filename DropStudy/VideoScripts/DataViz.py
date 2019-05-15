import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

real_vel = pd.read_csv('RealMapleData.csv',header=0,names=['trial','cutoff','velocity','rotation'])
norway_vel = pd.read_csv('NorwayMapleData.csv',header=0,names=['trial','cutoff','velocity','rotation'])
silver_vel = pd.read_csv('SilverMapleData.csv',header=0,names=['trial','cutoff','velocity','rotation'])

vels =list(real_vel['velocity'].values)
angs =list(real_vel['rotation'].values)
Rvelocities  = []; 
Rrotation  = []; 
for val in zip(vels,angs):
    if not ' ' in val:
        Rvelocities.append(float(val[0]))
        Rrotation.append(float(val[1]))

print('Real Norway maples \nMean: ',np.mean(Rvelocities),'\n','Std: ',np.std(Rvelocities),'\n\n')

plt.figure()
plt.hist(Rvelocities)
plt.figure()
plt.hist(Rrotation)

vels = list(norway_vel['velocity'].values)
angs =list(norway_vel['rotation'].values)
Nvelocities = []; 
Nrotation  = []; 
for val in zip(vels,angs):
    if not ' ' in val:
        Nvelocities.append(float(val[0]))
        Nrotation.append(float(val[1]))

print('3D-printed Norway maples \nMean: ',np.mean(Nvelocities),'\n','Std: ',np.std(Nvelocities),'\n\n')
plt.figure()
plt.hist(Nvelocities)
plt.figure()
plt.hist(Nrotation)

vels =list(silver_vel['velocity'].values)
angs =list(silver_vel['rotation'].values)
Svelocities  = []; 
Srotation = []; 
for val in zip(vels,angs):
    if not ' ' in val:
        Svelocities.append(float(val[0]))
        Srotation.append(float(val[1]))

print('3D-printed Silver maples \nMean: ',np.mean(Svelocities),'\n','Std: ',np.std(Svelocities),'\n\n')

plt.figure()
plt.hist(Svelocities)
plt.figure()
plt.hist(Srotation)
plt.show()

