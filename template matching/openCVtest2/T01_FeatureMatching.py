import cv2 as cv
import numpy as np

logo_small = cv.imread("T01/JTBC_logo_small.png")
logo_big = cv.imread("T01/JTBC_logo_big.png")

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(logo_small, None)
kp2, des2 = orb.detectAndCompute(logo_big, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

img_match = cv.drawMatches(logo_small, kp1, logo_big, kp2, matches[:10], outImg=None,
                          matchColor=(0, 255, 0), singlePointColor=None, flags=2)

cv.imshow("logo1", logo_small)
cv.waitKey(0)
cv.imshow("logo2", logo_big)
cv.waitKey(0)
cv.imshow("matched", img_match)
cv.waitKey(0)