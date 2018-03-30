import cv2
import numpy as np
from numpy import ma
import matplotlib
import matplotlib.pyplot as plt
import skimage.measure as skime
import skimage.morphology as skimo
import skimage.filters as skif
import os
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

fname = 'DSC_0061.jpg'

image = cv2.imread(fname, 0)

plt.imshow(image)
plt.show()
