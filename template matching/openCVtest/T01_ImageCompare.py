import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread("T01/background.jpg")
img2 = cv.imread("T01/background2.jpg")
img3 = cv.imread("T01/ScreenShot.jpg")
img4 = cv.imread("T01/ScreenShot2.jpg")

''' compare by subtraction and any()
    if difference have more than 1 of value, it becomes to false 
    it means different '''
difference1 = img1 - img2
# print(difference1)
difference1 = not difference1.any()

difference2 = img3 - img4
# print(difference2)
difference2 = not difference2.any()

if difference1:
    print("img1 and img2 are same")
else:
    print("img1 and img2 are different")

if difference2:
    print("img3 and img4 are same")
else:
    print("img3 and img4 are different")