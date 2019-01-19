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

def norm(image):
  # for m in range(len(image[0, :])):
  #   for n in range(len(image[:, 0])):
  #     if image[n, m] < 0.03
  #       image[n, m] = 0
  # image = image-np.min(image)
  binary = image > 0.03
  dilated = skimo.binary_dilation(binary, skimo.disk(2))
  cleaned = skimo.remove_small_objects(skimo.remove_small_holes(dilated))
  image = ma.array(image-np.min(image), mask=~cleaned).filled(0)
  if np.max(image) > 0:
    image = image/np.max(image)
  return image

def ind0(a):
  return a[0]

# thresh = skif.try_all_threshold(crop)
def findcentroid(image):
  image = image[:, 500:1050]
  binary = norm(image) > 0.01
  label, num = skime.label(binary, connectivity=2, return_num=True)
  if num > 0:
    props = [(p.area, p.centroid) for p in skime.regionprops(label)]
    if len(props)>1:
      props.sort(key=ind0, reverse=True)
    if props[0][0] > 100:
      return props[0]


def saveNorm(image):
  image = norm(image)[:]
  image.dtype
  data = (image*255).astype('uint8')
  data.dtype
  cv2.imwrite('../'+base+'-norm/'+fname, data)

# fname = 'r-g05-t01-f01691.tif'

# image = tf.imread(fname)
# crop = image[:, 600:1000, 0]
# base = 'testvid'

# binary = norm(crop) > 0.2

# plt.imshow(binary)

# label, num = skime.label(binary, return_num=True)
# props = skime.regionprops(label)
os.chdir('2017July07')
for file in os.listdir('.'):
  if file[1] == '-':
    base = file
    os.chdir(base)
    for file in os.listdir('.'):
      if file[-4:] == '.avi':
        vidname = file
    video = cv2.VideoCapture(vidname)
    with open('../../DropData/'+base[:9]+'-data.csv', 'w') as data:
      data.write('FrameNo, X, Y\n')
      ok, frame = video.read()
      bgimage = frame[:, :, 0]
      # bgimage = cv2.imread(base + '-00001.tif', 0)
      # centroids = []
      # framenums = []
      # for a in range(1, len(os.listdir('.'))+1):
      for a in range(1, round(video.get(7))):
        # fname = base + '-{:05}.tif'.format(a)
        # if fname not in os.listdir('.'):
          # print(fname)
        # else:
          # image = cv2.imread(fname, 0)
        ok, frame = video.read()
        if not ok:
          print(vidname+' frame no {} not loaded'.format(round(video.get(1))))
        else:
          image = frame[:, :, 0]
          props = findcentroid(image/256-bgimage/256)
          if not props == None:
            data.write('{}, {}, {} \n'.format(a, props[1][1], props[1][0]))
            # framenums.append(a)
            # centroids.append(props[1])
            # print(a, ' ', props[1], ' area=', props[0])
          # saveNorm(image/256-bgimage/256)
    os.chdir('..')
os.chdir('..')
# C = np.array(centroids)

# Y = C[100:, 0]
# X = C[100:, 1]-np.mean(C[100:, 1])
# T = np.array(framenums)[100:]
# out = np.fft.fft(X)
# mag = np.sqrt(out*out)
# freqs = np.fft.fftfreq(out.size, d=1/2000)

# plt.figure(1, figsize=(8, 8))
# plt.plot(X, Y)
# plt.gca().invert_yaxis()
# plt.xlabel('$x$', fontsize=28)
# plt.ylabel('$y$', fontsize=28)

# plt.figure(2, figsize=(8, 8))
# plt.plot(T, Y)
# plt.gca().invert_yaxis()
# plt.xlabel('Time(frames)', fontsize=28)
# plt.ylabel('$y$', fontsize=28)

# plt.figure(3, figsize=(8, 8))
# plt.plot(T, X)
# plt.xlabel('Time(frames)', fontsize=28)
# plt.ylabel('$x$', fontsize=28)

# for i in plt.get_fignums():
#   plt.figure(i)
#   plt.gca().set_aspect('equal', adjustable='box', anchor='C')

# print('Angular velocity: ', freqs[np.argmax(mag)]*60, ' rpm')