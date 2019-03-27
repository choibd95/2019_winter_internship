import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

logo = cv.imread("T07/twitch_logo.png")
img = cv.imread("T07/twitch_image_window.jpg")

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(logo, None)
kp2, des2 = orb.detectAndCompute(img, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

img_out = cv.imread("T07/twitch_image_window.jpg")
img_match = cv.drawMatches(logo, kp1, img, kp2, matches[:10], img_out, -1, -1)

plt.imshow(img_match), plt.show()