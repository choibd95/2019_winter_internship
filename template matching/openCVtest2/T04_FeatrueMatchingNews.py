import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

logo = cv.imread("T04/JTBC_logo.jpg")
video = cv.VideoCapture("T04/JTBC_news.mp4")

orb = cv.ORB_create()

kp_logo, des_logo = orb.detectAndCompute(logo, None)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

while True:

    ret, frame = video.read()

    if not ret:
        break

    height, width, channel = frame.shape
    pts1 = np.float32([[width / 4 * 3, 0],
                       [width / 4 * 3, height / 4],
                       [width, 0],
                       [width, height / 4]])
    pts2 = np.float32([[0, 0],
                       [0, height / 2],
                       [width / 3, 0],
                       [width / 3, height / 2]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    frame = cv.warpPerspective(frame, M, (int(width / 2), int(height / 2)))

    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    match = bf.match(des_logo, des_frame)
    match = sorted(match, key=lambda x: x.distance)

    img_match = cv.drawMatches(logo, kp_logo, frame, kp_frame, match,
                               outImg=None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)

    cv.imshow("matched", img_match)
    cv.waitKey(100)