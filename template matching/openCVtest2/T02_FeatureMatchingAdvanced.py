import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

logo = cv.imread("T02/JTBC_logo.jpg")
broadcasting = []
broadcasting.append(cv.imread("T02/JTBC_broadcasting1.jpg"))
broadcasting.append(cv.imread("T02/JTBC_broadcasting2.jpg"))
broadcasting.append(cv.imread("T02/JTBC_broadcasting3.jpg"))

height = [0]*3
width = [0]*3

for i in range(3):
    height[i], width[i], channel = broadcasting[i].shape
    #print(height[i]/2.0, width[i]/2.0)
    pts1 = np.float32([[width[i]/4*3,   0],
                       [width[i]/4*3,   height[i]/4],
                       [width[i],     0],
                       [width[i],     height[i]/4]])
    pts2 = np.float32([[0,            0],
                       [0,            height[i]/2],
                       [width[i]/3,     0],
                       [width[i]/3,     height[i]/2]])

    '''
    cv.circle(broadcasting[i], (int(width[i]/4*3), 0), 10, (255, 0, 0), -1)
    cv.circle(broadcasting[i], (int(width[i]/4*3), int(height[i]/4)), 10, (0, 255, 0), -1)
    cv.circle(broadcasting[i], (width[i], 0), 10, (0, 0, 255), -1)
    cv.circle(broadcasting[i], (width[i], int(height[i]/4)), 10, (0, 0, 0), -1)

    cv.imshow("circle", broadcasting[i])
    cv.waitKey(0)
    '''

    M = cv.getPerspectiveTransform(pts1, pts2)
    broadcasting[i] = cv.warpPerspective(broadcasting[i], M, (int(width[i]/2), int(height[i]/2)))



orb = cv.ORB_create()

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
match = [0]*3
img_match = [0]*3

kp_logo, des_logo = orb.detectAndCompute(logo, None)
kp_b = [0]*3
des_b = [0]*3

for i in range(3):
    kp_b[i], des_b[i] = orb.detectAndCompute(broadcasting[i], None)
    match[i] = bf.match(des_logo, des_b[i])
    match[i] = sorted(match[i], key=lambda x: x.distance)
    img_match[i] = cv.drawMatches(logo, kp_logo, broadcasting[i], kp_b[i], match[i],
                                  outImg=None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)
    cv.imshow("matching", img_match[i])
    cv.waitKey(0)

