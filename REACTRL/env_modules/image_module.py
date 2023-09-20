import cv2
import numpy as np
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt

def image_process(img, kernal_v=6, show_fig=True) -> tuple:
    blur = cv2.GaussianBlur(img, (5, 5), 2)
    h, w = img.shape[:2]
    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernal_v, kernal_v))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lowerb, upperb)
    for row in range(h):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, w-1] == 255:
            cv2.floodFill(binary, None, (w-1, row), 0)

    for col in range(w):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[h-1, col] == 255:
            cv2.floodFill(binary, None, (col, h-1), 0)

    foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    background = cv2.dilate(foreground, kernel, iterations=3)
    unknown = cv2.subtract(background, foreground)

    # Convert the image to grayscale
    gray = background

    background[background==255] = 5
    background[background==0] = 255
    background[background==5] = 0

    # Apply a threshold to the image to
    # separate the objects from the background
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find the contours of the objects in the image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and calculate the area of each object

    figure=plt.figure()
    fig, ax = plt.subplots()
    ellipses=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Draw a bounding box around each
        # object and display the area on the image
        x, y, w, h = cv2.boundingRect(cnt)
        xy, width_height, angle = cv2.fitEllipse(cnt)
        ellipses.append([xy[0], xy[1], width_height[0], width_height[1], angle])
        if show_fig:
            cv2.ellipse(img, (int(xy[0]), int(xy[1])), (int(width_height[0]/2), int(width_height[1]/2)), angle, 0, 360, (0, 255, 255), 2)
            cv2.ellipse(background, (int(xy[0]), int(xy[1])), (int(width_height[0]/2), int(width_height[1]/2)), angle, 0, 360, (100, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(img, str(area), (x, y+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.rectangle(background, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(background, str(area), (x, y+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 144, 100), 2)
    if show_fig:
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(background)
    return ellipses
    