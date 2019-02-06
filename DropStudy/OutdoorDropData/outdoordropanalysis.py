import pandas as pd
import numpy as np

data = pd.read_excel('2017-04 Samara Drop Tests.xlsx', header=1)

samara_list = ['SM', 'NM', 'N']
prop_list = ['Time', 'Dist', 'Ang']
for index, row in data.iterrows():
  print(row)
  if index == 4:
    print(row['Time - NM'])
  for samara_type in samara_list:
    for prop in prop_list:
      val = row[prop + ' - ' + samara_type]
      if val == '-':
        print('No Data Recorded')
      elif ~np.isnan(val):
        print(index, prop, samara_type, val)
