#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy
import collections
import numpy as np

class VCTracker(object):

    def __init__(self, image, region):
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        #Initial template
        self.template = image[int(top):int(bottom), int(left):int(right)]
        #Center position of the template (u,v)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        #Size of the template (width, height)
        self.size = (region.width, region.height)
        
        #Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        im = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255,0,0), 2)
        cv2.imshow('result',im)
        cv2.imshow('template',self.template)
        cv2.waitKey(1) #change 0 to 1 - remove waiting for key press


    # *******************************************************************
    # This is the function to fill. You can also modify the class and add additional
    # helper functions and members if needed
    # It should return, in this order, the u (col) and v (row) coordinates of the top left corner
    # the width and the height of the bounding box
    # *******************************************************************
    def track(self, image):

	# Fill here the function
	# You have the information in self.template, self.position and self.size
	# You can update them and add other variables
        left = 0
        top = 0
        confidence = 0
	
        
        # Fill here the function
        res = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        
        img_downscaled = self.resize_img(image,factor=0.75)
        res_down = cv2.matchTemplate(img_downscaled, self.template, cv2.TM_CCORR_NORMED)
        
        img_upscaled = self.resize_img(image,factor=1.25)
        res_up = cv2.matchTemplate(img_upscaled, self.template, cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        best_loc = max_loc
        best_val = max_val

        img_width = image.shape[0]
        img_height = image.shape[1]

        self.position = (best_loc[0] + self.size[0] / 2.0, best_loc[1] + self.size[1] / 2.0)
        self.mask = self.create_circular_mask(img_height,img_width,center=self.position, radius=self.size[0] * 1.75)
        self.masked_img = image.copy()
        self.masked_img[~self.mask] = 0

        #Downscale image to look for bigger object
        self.img_downscaled = self.resize_img(image,factor=0.75)

        #Upscale image to look for smaller object
        self.img_upscaled = self.resize_img(image, factor=1.25)

        #return best_loc[0], best_loc[1], self.w, self.h
        left = best_loc[0]
        top = best_loc[1]
        return vot.Rectangle(left, top, self.size[0], self.size[1]), confidence


    #Creates a circular mask centered and with a radius defined.
    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:w, :h]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    
    
    def resize_img(self,image, factor=0.75):
        if factor > 1:
            #Uses interpolation recommended to enlarge image
            image = cv2.resize(image,(0,0),fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
        else:
            #Uses interpolation recommended to reduce image
            image  = cv2.resize(image,(0,0),fx=factor,fy=factor,interpolation=cv2.INTER_AREA)

        img_width = image.shape[0]
        img_height = image.shape[1]

        mask = self.create_circular_mask(img_height,img_width,center= np.array(self.position)*factor, radius=self.size[0] * 1.75)

        image[~mask] = 0
        return image

        
        
# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************
handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
image = cv2.imread(imagefile)

# Initialize the tracker
tracker = VCTracker(image, selection)

while True:
    # *****************************************
    # VOT: Call frame method to get path of the
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    
    # Track the object in the image  
    region, confidence = tracker.track(image)
    
    #Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit
    im = cv2.rectangle(image,(int(region.x),int(region.y)),(int(region.x+region.width),int(region.y+region.height)), (255,0,0), 2)
    cv2.imshow('result',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)

