import cv2 as cv
import numpy as np


black = np.zeros((100, 100), np.uint8)
while True:
    cv.imshow("aa", black)
    oper = cv.waitKey(500)
    if oper:
        print(oper)
    oper = None
