import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

video_full = cv.VideoCapture("T05/news_full.mp4")
video_frag = cv.VideoCapture("T05/news_fragment.mp4")

orb = cv.ORB_create()

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

while True:

    ret_full, frame_full = video_full.read()
    ret_frag, frame_frag = video_frag.read()

    if not ret_full or ret_frag:
        break

    '''
    height, width, channel = frame_full.shape
    pts1 = np.float32([[width / 3 * 2, 0],
                       [width / 3 * 2, height / 3],
                       [width, 0],
                       [width, height / 3]])
    pts2 = np.float32([[0, 0],
                       [0, height / 2],
                       [width / 2, 0],
                       [width / 2, height / 2]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    frame_full = cv.warpPerspective(frame_full, M, (int(width / 2), int(height / 2)))
    '''
    kp_frame_full, des_frame_full = orb.detectAndCompute(frame_full, None)
    kp_frame_frag, des_frame_frag = orb.detectAndCompute(frame_frag, None)

    match = bf.match(des_frame_frag, des_frame_full)
    match = sorted(match, key=lambda x: x.distance)

    img_match = cv.drawMatches(frame_frag, kp_frame_frag, frame_full, kp_frame_full, match,
                               outImg=None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)

    cv.imshow("matched", img_match)
    cv.waitKey(100)