import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread("T02/background_big.jpg")
img2 = cv.imread("T02/background_small.jpg")

''' use resize to compare pictures with different size
    it can make same size but result is not good for expectation '''
img1_A = cv.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

difference = img1_A - img2
# print(difference)
difference = not difference.any()

if difference:
    print("img1 and img2 are same")
else:
    print("img1 and img2 are different")

'''
cv.imshow("img1", img1_A)
cv.waitKey(0)
cv.imshow("img2", img2)
cv.waitKey(0)
'''