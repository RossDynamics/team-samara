import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skfilt
import skimage.morphology as skmorph
import skimage.measure as skmeas
import skimage.segmentation as skseg

### Defines a square search area to look for samara
### based on the previous centroid location
def searchArea(centroid, size=150):
  # [rowmin, rowmax, colmin, colmax]
  rowmin = max(int(round(centroid[0]))-size//2, 0)
  colmin = max(int(round(centroid[1]))-size//2, 0)
  return [rowmin, rowmin+size, colmin, colmin+size]

def videorun(filename, savefile,saevfile2):
  with open(savefile, 'w') as data:
    data.write('FrameNo,Row,Column,Angle\n')
    video = cv2.VideoCapture(filename)

    ### Average first n_avg frames in the top initsize[0] rows of the image
    initsize = (150,1280)
    n_avg = 40
    vidavg = np.zeros(initsize)
    for k in range(n_avg):
      ok, frame = video.read()
      if ok:
        vidavg += frame[:initsize[0], :initsize[1], 0].astype('float')/n_avg
#    ok, img = video.read()
#    cv2.imwrite(savefile2+'_frame_image.png',img)
    ### Subtracting vidavg, we wait to identify the first object that appears
    ### in the top initsize[0] rows of the image
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
        clean = skmorph.remove_small_objects(thresh, min_size=150) 
        labeled = skseg.clear_border(skmorph.label(clean))
        ### When an object appears, save its properties and break out of loop (first=True)
        if np.max(labeled) > 0:
          props = [(p.area, p.centroid, p.orientation) for p in skmeas.regionprops(labeled)]
          props.sort(reverse=True)
          cent = props[0][1]
          # output.append((cent,props[0][2]))
          data.write('{},{},{},{}\n'.format(int(round(video.get(1))), cent[0], cent[1], props[0][2]))
          lastcent = cent
          first = True
    
    ### Rest of loop.
    fail = 0
    while ok:
      ok, frame = video.read()
      if ok:
#        fig, ax = plt.subplots(1,2)
        window = searchArea(lastcent)
        image = frame[window[0]:window[1], window[2]:window[3], 0]
        high = skfilt.threshold_yen(image)
        low = high-(high-np.min(image))//4
        thresh = skfilt.apply_hysteresis_threshold(image, low, high)
        # plt.sca(ax[0])
        # plt.imshow(image > high)
        # plt.sca(ax[1])
        # plt.imshow(thresh)
        # plt.show()
        clean = skmorph.remove_small_objects(thresh, min_size=50)
        dilated = skmorph.binary_dilation(thresh, selem=skmorph.square(3))
        labeled = skseg.clear_border(skmorph.label(dilated))
        if np.max(labeled) > 0:
          props = [(p.area, p.centroid, p.orientation) for p in skmeas.regionprops(labeled)]
          props.sort(reverse=True)
          cent = [props[0][1][k]+window[2*k] for k in range(2)]
          # output.append((cent,props[0][2]))
          data.write('{},{},{},{}\n'.format(int(round(video.get(1))), cent[0], cent[1], props[0][2]))
          lastcent = cent
          fail = 0
#          plt.figure(int(round(video.get(1))))
#          plt.imshow(thresh)
#          
        ### Break loop when no objects detected
#        else:
#            ok = False
        elif fail > 4:
            ok = False
        else:
            fail += 1
#            plt.figure(int(round(video.get(1))))
#            plt.imshow(thresh)
            


### Main loop
os.chdir('2017July07')
#os.chdir('White Background')
for file in os.listdir('.'):
#   if file[:3] == 'n-g':
  if file[:9] == 'n-g30-t03':
    base = file
    os.chdir(base)
    for file in os.listdir('.'):
      if file[-4:] == '.avi':
        vidname = file
    savefile = '../../DropData_temp/'+base[:9]+'-data.csv'
    savefile2 = '../../DropData_temp/'+base[:9]
    videorun(vidname,savefile,savefile2)
    os.chdir('..')
#   if file[:3] == 'r-g':
##  if file[:9] == 'n-g08-t01':
#    base = file
#    os.chdir(base)
#    for file in os.listdir('.'):
#      if file[-4:] == '.avi':
#        vidname = file
#    savefile = '../../DropData3/'+base[:9]+'-data.csv'
#    savefile2 = '../../DropData3/'+base[:9]
#    videorun(vidname,savefile,savefile2)
#    os.chdir('..')
#   if file[:3] == 's-g':
##  if file[:9] == 'n-g08-t01':
#    base = file
#    os.chdir(base)
#    for file in os.listdir('.'):
#      if file[-4:] == '.avi':
#        vidname = file
#    savefile = '../../DropData3/'+base[:9]+'-data.csv'
#    savefile2 = '../../DropData3/'+base[:9]
#    videorun(vidname,savefile,savefile2)
#    os.chdir('..')
os.chdir('..')