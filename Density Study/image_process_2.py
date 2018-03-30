# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:49:11 2018

@author: Nathaniel
"""

# import the necessary packages
from __future__ import print_function
from scipy.spatial import distance as dist
import numpy as np
import cv2
from imutils import perspective
from imutils import contours
import argparse
import imutils
import matplotlib.pyplot as plt
import skimage.morphology as skimo
import numpy.ma as ma
from PIL import Image
 
def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, default=-1,
	help="whether or not the new order points should should be used")
args = vars(ap.parse_args())
 
def norm(image):

  # for m in range(len(image[0, :])):

  #   for n in range(len(image[:, 0])):

  #     if image[n, m] < 0.03

  #       image[n, m] = 0

  # image = image-np.min(image)

  binary = image > 0.2

  dilated = skimo.binary_dilation(binary, skimo.disk(2))

  cleaned = skimo.remove_small_objects(skimo.remove_small_holes(dilated), min_size=800000)

  image = ma.array(image-np.min(image), mask=~cleaned).filled(0)

  if np.max(image) > 0:

    image = image/np.max(image)

  return image


img = Image.new('RGB', (4000, 6000), (139, 770, 40))
# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread("DSC_0061.jpg")
sub = image-img
sub = sub[:,:,2]-sub[:,:,0]
#image = image[:,2000:,:]
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sub = cv2.line(sub,(0,0),(4000,0),(0,0,0),200)
sub = cv2.line(sub,(0,6000),(4000,6000),(0,0,0),200)
#image2 = image[:, :, 2]-image[:,:,0]
#gray = cv2.erode(image,None,iterations=10)
gray = sub/256
gray1 = norm(gray)
gray2 = cv2.GaussianBlur(gray1, (27, 27), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
#imageio.imwrite('temp_gray.jpg', gray2)
#gray3 = cv2.imread('temp_gray.jpg',0)
gray3 = np.uint8(gray2*256)
edged = cv2.Canny(gray3, 0, 20,apertureSize = 3)
edged = cv2.dilate(edged, None, iterations=8)
edged = cv2.erode(edged, None, iterations=4)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over the contours individually
for (i, c) in enumerate(cnts):
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 10000:
		continue
 
	# compute the rotated bounding box of the contour, then
	# draw the contours
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	cv2.drawContours(image, [box], -1, (0, 255, 0), 10)
 
	# show the original coordinates
	print("Object #{}:".format(i + 1))
	print(box)
    
    	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	rect = order_points_old(box)
 
	# check to see if the new method should be used for
	# ordering the coordinates
	if args["new"] > 0:
		rect = perspective.order_points(box)
 
	# show the re-ordered coordinates
	print(rect.astype("int"))
	print("")
    
    	# loop over the original points and draw them
	for ((x, y), color) in zip(rect, colors):
		cv2.circle(image, (int(x), int(y)), 5, color, -1)
 
	# draw the object num at the top-left corner
	cv2.putText(image, "Object #{}".format(i + 1),
		(int(rect[0][0] - 15), int(rect[0][1] - 15)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
 
# show the image
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 10))
plt.imshow(image[:,:,[2,1,0]])
#	cv2.imshow("Image", image)
#cv2.waitKey(0)