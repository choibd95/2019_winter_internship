import numpy as np
import cv2 as cv

# add name of file
video = cv.VideoCapture("T05/twitch_logo_moving2-1.mp4")
logo = cv.imread("T05/twitch_logo2-1.jpg", 0)

firstFrame = None
count = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = logo.shape[::-1]
    res = cv.matchTemplate(frame_gray, logo, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

    cv.imshow("matching", frame)
    cv.waitKey(30)
