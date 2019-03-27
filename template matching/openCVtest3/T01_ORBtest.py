import cv2 as cv
import numpy as np

# 최소 matching 횟수, 정확도
minimumMatchPoint = 10
ratio = 0.7

# 이미지 로드
# gray scale로 읽어와도, None으로 그대로 읽어와도 무방함
logo = cv.imread("T01/CocaCola_logo.png")
img = cv.imread("T01/CocaCola_can.png")

img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

# ORB 객체 생성, 각 리소스 별 특징점 탐색
orb = cv.ORB_create()
kp_logo, des_logo = orb.detectAndCompute(logo, None)
kp_img, des_img = orb.detectAndCompute(img, None)

# FLANN matcher 파라미터 정의
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=10)

# FLANN matcher 정의, match 결과 저장
flann = cv.FlannBasedMatcher(index_params, search_params)
match_flann = flann.knnMatch(des_logo, des_img, k=2)
#print("match_flann: ", match_flann)

# 결과 내에서 가장 적합한 값만을 선택
good_flann = []
for p, q in match_flann:
    if p.distance < q.distance*ratio:
        good_flann.append(p)
#print("good: ", good)

#ifdef 1
if len(good_flann) > minimumMatchPoint:
    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_flann]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_flann]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchMask = mask.ravel().tolist()

    height, width, channel = logo.shape
    pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    img = cv.polylines(img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    print("no enough matches are found")
    matchMask = None
#endif
#matchMask = None

# 선별한 matching 결과를 image 형태로 쓰고 출력
img_match = np.empty((max(logo.shape[0], img.shape[0]), logo.shape[1] + img.shape[1], 3), dtype=np.uint8)
cv.drawMatches(logo, kp_logo, img, kp_img, good_flann,
               outImg=img_match, matchColor=None, singlePointColor=(255, 255, 255), matchesMask=matchMask, flags=2)
cv.imshow("flann matching", img_match)
cv.waitKey(0)

'''
# BF matcher 정의, match 결과 저장
bf = cv.BFMatcher()
match_bf = bf.match(des_logo, des_img)

print("match_bf: ", match_bf)

# 선별한 matching 결과를 image 형태로 쓰고 출력
# matched keypoint가 너무 많아 30개로 제한, 배열 범위 변경으로 조정 가능
img_match = np.empty((max(logo.shape[0], img.shape[1]), logo.shape[1] + img.shape[1], 3), dtype=np.uint8)
cv.drawMatches(logo, kp_logo, img, kp_img, match_bf[:30],
               outImg=img_match, matchColor=None, singlePointColor=(255, 255, 255), flags=2)
cv.imshow("bf matching", img_match)
cv.waitKey(0)
'''
