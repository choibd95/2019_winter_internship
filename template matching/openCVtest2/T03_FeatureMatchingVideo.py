import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

logo = cv.imread("T03/JTBC_logo.jpg")
video = cv.VideoCapture("T03/JTBC_logo_changesize.mp4")
logo = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()

kp_logo, des_logo = orb.detectAndCompute(logo, None)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    #match = flann.knnMatch(des_logo, des_frame, k=2)
    match = bf.match(des_logo, des_frame)
    match = sorted(match, key=lambda x: x.distance)

    img_match = cv.drawMatches(logo, kp_logo, frame, kp_frame, match,
                               outImg=None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)

    cv.imshow("matched", img_match)
    cv.waitKey(30)