
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skfilt
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.segmentation as skseg


def searchArea(centroid, size=200):
  # [rowmin, rowmax, colmin, colmax]
  rowmin = max(int(round(centroid[0]))-size//2, 0)
  colmin = max(int(round(centroid[1]))-size//2, 0)
  return [rowmin, rowmin+size, colmin, colmin+size]


def videorun(filename, savefile):
  with open(savefile, 'w') as data:
    data.write('FrameNo,Row,Column,Angle\n')
    video = cv2.VideoCapture(filename)
    initsize = (150,1280)
    n_avg = 40
    vidavg = np.zeros(initsize)
    for k in range(n_avg):
      ok, frame = video.read()
      if ok:
        vidavg += frame[:initsize[0], :initsize[1], 0].astype('float')/n_avg

    video = cv2.VideoCapture(filename)
    first = False
    while not first:
      ok, frame = video.read()
      if ok:
        top = frame[:initsize[0], :initsize[1], 0].astype('float')-vidavg
        top[top <= 0.0] = 0.0
        # plt.hist(top.ravel(), 100)
        yen = skfilt.threshold_yen(top)
        thresh = top > yen
        clean = skmorph.remove_small_objects(thresh, min_size=80) 
        labeled = skseg.clear_border(skmorph.label(clean))
        if np.max(labeled) > 0:
          props = [(p.area, p.centroid, p.orientation) for p in skmeas.regionprops(labeled)]
          props.sort(reverse=True)
          cent = props[0][1]
          # output.append((cent,props[0][2]))
          data.write('{},{},{},{}\n'.format(int(round(video.get(1))), cent[0], cent[1], props[0][2]))
          lastcent = cent
          first = True

    # k = 0
    while ok:
    # for k in range(8):
      ok, frame = video.read()
      if ok:
        fig, ax = plt.subplots(1,2)
        window = searchArea(lastcent)
        image = frame[window[0]:window[1], window[2]:window[3], 0]
        high = skfilt.threshold_yen(image)
        low = high-(high-np.min(image))//3
        thresh = skfilt.apply_hysteresis_threshold(image, low, high)
        # plt.sca(ax[0])
        # plt.imshow(image > high)
        # plt.sca(ax[1])
        # plt.imshow(thresh)
        # plt.show()
        clean = skmorph.remove_small_objects(thresh, min_size=80)
        dilated = skmorph.binary_dilation(thresh, selem=skmorph.square(3))
        labeled = skseg.clear_border(skmorph.label(dilated))
        if np.max(labeled) > 0:
          props = [(p.area, p.centroid, p.orientation) for p in skmeas.regionprops(labeled)]
          props.sort(reverse=True)
          cent = [props[0][1][k]+window[2*k] for k in range(2)]
          # output.append((cent,props[0][2]))
          data.write('{},{},{},{}\n'.format(int(round(video.get(1))), cent[0], cent[1], props[0][2]))
          lastcent = cent
        else: ok = False
          # k += 1

os.chdir('2017July07')
for file in os.listdir('.'):
  # if file[1] == '-':
  if file[:9] == 'n-g02-t02':
    base = file
    os.chdir(base)
    for file in os.listdir('.'):
      if file[-4:] == '.avi':
        vidname = file
    savefile = '../../DropData2/'+base[:9]+'-data.csv'
    videorun(vidname,savefile)
    os.chdir('..')
os.chdir('..')