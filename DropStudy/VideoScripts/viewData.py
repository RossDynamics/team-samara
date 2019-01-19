import os
import pandas as pd
import matplotlib.pyplot as plt

for file in os.listdir('DropData'):
  drop = pd.read_csv('DropData/'+file)
  plt.plot(drop[' X'], drop[' Y'])
  plt.title(file)
  plt.gca().invert_yaxis()
  plt.show()
