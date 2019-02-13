import os
import pandas as pd
import matplotlib.pyplot as plt

for file in os.listdir('DropData_temp'):
  drop = pd.read_csv('DropData_temp/'+file)
  plt.plot(drop['Row'], drop['Column'])
  plt.title(file)
  plt.gca().invert_yaxis()
  plt.legend(('1','2','3','1','2','3'))
  plt.show()
