#!/usr/bin/python

import vot
import sys
import time
import cv2
import numpy
import collections


class VCTracker(object):

    def __init__(self, image, region):
        # Tocar la escala de la imagen
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        # Initial template
        self.template = image[int(top):int(bottom), int(left):int(right)]
        # Center position of the template (u,v)
        self.position = (region.x + region.width / 2,
                         region.y + region.height / 2)
        # Size of the template (width, height)
        self.size = (region.width, region.height)

        # Use these lines for testing.
        # Comment them when you evaluate with the vot toolkit
        im = cv2.rectangle(image, (int(left), int(top)),
                           (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.imshow('result', im)
        cv2.imshow('template', self.template)
        cv2.waitKey(1)  # change 0 to 1 - remove waiting for key press

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

        # MODIFICACION O MEJORAS
        # -Si es una video, coger el template del frame antiguo por si cambia de posicion

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(
            round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(
            round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        # Si nuestra imagen es mas pequeña que el template que tenemos, el rectangulo sera la imagen completa
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(
            cut, self.template, cv2.TM_CCORR_NORMED)  # TM_CCORR_NORMED
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        self.position = (left + max_loc[0] + float(self.size[0]) /
                         2, top + max_loc[1] + float(self.size[1]) / 2)

        return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]), max_val

    def changeTemplate(self, region,confidence, image):
        # Cambiamos el template en caso de que el objeto que estamos siguiendo se mueva y cambie de posicion con respecto al template inicial
            left = max(region.x, 0)
            top = max(region.y, 0)

            right = min(region.x + region.width, image.shape[1] - 1)
            bottom = min(region.y + region.height, image.shape[0] - 1)
            tracker.template = image[int(top):int(
            bottom), int(left):int(right)]


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
    # Cambiamos el template en caso de que el objeto que estamos siguiendo se mueva y cambie de posicion con respecto al template inicial
    tracker.changeTemplate(region,confidence,image)

    # *****************************************
    # VOT: Create VOT handle at the beginning
    #      Then get the initializaton region
    #      and the first image
    # *****************************************
    # Use these lines for testing.
    # Comment them when you evaluate with the vot toolkit

    im = cv2.rectangle(image, (int(region.x), int(region.y)), (int(
        region.x+region.width), int(region.y+region.height)), (255, 0, 0), 2)
    cv2.imshow('result', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)
