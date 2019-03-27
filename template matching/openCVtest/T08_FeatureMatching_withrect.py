import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

logo = cv.imread("T08/twitch_logo.png")
img = cv.imread("T08/twitch_image_window.jpg")

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(logo, None)
kp2, des2 = orb.detectAndCompute(img, None)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)

fla = cv.FlannBasedMatcher(index_params, search_params)

matches = fla.knnMatch(des1, des2, k=2)

good = []

for m in matches:
    #if m.distance < 0.7*n.distance:
    good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w, d = img.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    logo = cv.polylines(logo, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    print("not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img_match = None
img_match = cv.drawMatches(logo, kp1, img, kp2, good, img_match, flags=2)

plt.imshow(img_match), plt.show()
