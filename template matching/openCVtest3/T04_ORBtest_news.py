import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 4개의 좌표가 모든 내각이 음각인 사각형을 이루는 지 확인하기 위한 함수
# 대각선의 교점이 사각형 내에 존재하는지 검사, 결과를 반환
def crossPointCheck(sq):

    # 연립일차방정식 계수행렬의 행렬식 계산
    def det(A, B):
        return A[0] * B[1] - A[1] * B[0]

    # 주어진 좌표로부터 크래머 공식 적용을 위한 일차방정식 계수 계산
    # a : x의 계수
    # b : y의 계수
    # D : 상수
    def lineCoefficient(line):
        a = line[1][1] - line[0][1]
        b = line[0][0] - line[1][0]
        D = det(line[0], line[1])
        return [a, b, D]

    # 결과로 도출한 좌표가 사각형 내에 존재하는지 검사, 결과를 반환
    # 꼭짓점 좌표와 결과 좌표를 배열로 저장, 해당 배열이 오름차순 또는 내림차순인지를 검사
    # p, q  : 대각선 선분 양 끝 좌표값
    # pos   : 결과 도출 좌표값
    def isInSquare(p, q, pos):
        px = [p[0][0], pos[0], p[1][0]]
        py = [p[0][1], pos[1], p[1][1]]
        qx = [q[0][0], pos[0], q[1][0]]
        qy = [q[0][1], pos[1], q[1][1]]

        in_here = True

        if not ((all(px[i] < px[i+1] for i in range(len(px)-1)) or
                 all(px[i] > px[i+1] for i in range(len(px)-1))) and
                (all(qx[i] < qx[i+1] for i in range(len(qx)-1)) or
                 all(qx[i] > qx[i+1] for i in range(len(qx)-1)))):
            in_here = False
        if not ((all(py[i] < py[i+1] for i in range(len(py)-1)) or
                 all(py[i] > py[i+1] for i in range(len(py)-1))) and
                (all(qy[i] < qy[i+1] for i in range(len(qy)-1)) or
                 all(qy[i] > qy[i+1] for i in range(len(qy)-1)))):
            in_here = False

        return in_here

    # 파라미터로 받은 좌표를 선분 형태로 저장
    # 선분을 포함하는 직선의 일차방정식 계수 계산 및 저장
    line1 = [sq[0][0], sq[2][0]]
    line2 = [sq[1][0], sq[3][0]]
    lc1 = lineCoefficient(line1)
    lc2 = lineCoefficient(line2)

    # x, y 계수 값으로부터 행렬식 계산
    # 해당 행렬식은 해(교점)의 존재유무를 판단 가능
    DET = det([lc1[0], lc1[1]], [lc2[0], lc2[1]])
    if DET == 0:
        return False

    # 크래머 공식에 따른 교점 좌표 계산
    x = det([lc1[2], lc1[1]], [lc2[2], lc2[1]]) / DET
    y = det([lc1[0], lc1[2]], [lc2[0], lc2[2]]) / DET

    # 사각형 내부의 해당 좌표 존재 여부를 판단 및 반환
    return isInSquare(line1, line2, [x, y])


# minimum_match     : 대상을 인식하는 최소 매칭 횟수
# ratio             : KLANN 매칭 비교 계수, 작을 수록 정확
# frame_counter, match_counter : 프레임 단위 인식률 측정 변수
minimum_match = 10
ratio = 0.8
frame_counter = 0
match_counter = 0

# logo, news를 읽어올 배열 선언
# 각각의 파일을 읽고 append
# index를 통한 test 대상 지정
logo_pool = []
logo_pool.append(cv.imread("T04/SBS_logo_2.png"))
logo_pool.append(cv.imread("T04/KBS_logo.png"))
logo_pool.append(cv.imread("T04/MBC_logo.png"))
logo_pool.append(cv.imread("T04/JTBC_logo.jpg"))

news_pool = []
news_pool.append(cv.VideoCapture("T04/SBS.mp4"))
news_pool.append(cv.VideoCapture("T04/KBS.mp4"))
news_pool.append(cv.VideoCapture("T04/MBC(360p).mp4"))
news_pool.append(cv.VideoCapture("T04/JTBC.mp4"))

logo = logo_pool[3]
news = news_pool[3]


# logo size 조정, 가우시안 블러 처리한 결과를 감산하여 image sharpening
logo = cv.resize(logo, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
# logo = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# logo = cv.filter2D(logo, -1, kernel)
logo_temp = cv.GaussianBlur(logo, (0, 0), 3)
logo = cv.addWeighted(logo, 2.0, logo_temp, -1.0, 0)



logo = cv.Canny(logo, 100, 200)
cv.imshow("logo edge", logo)
cv.waitKey(0)



# ORB 객체 선언, logo의 특징점 탐색 및 저장
orb = cv.ORB_create()
kp_logo, des_logo = orb.detectAndCompute(logo, None)

# FLANN matcher 파라미터 정의 및 객체 선언
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=200)
flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    # news로부터 frame 읽어옴
    # 읽지 못했을 경우, 영상의 끝부분 예외처리
    ret, frame = news.read()
    if frame is None:
        print("frame is none")
        break
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # frame 내 logo가 있을 것으로 예상되는 부분을 확대 및 frame size 변경
    height, width, c = frame.shape
    pts1 = np.float32([[width/6*5, 0],
                       [width/6*5, height/6],
                       [width, 0],
                       [width, height/6]])
    pts2 = np.float32([[0, 0],
                       [0, height/3],
                       [width/3, 0],
                       [width/3, height/3]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    frame = cv.warpPerspective(frame, M, (int(width/3), int(height/3)))
    # cv.imshow("cut", frame)

    # frame 가우시안 블러 처리한 결과를 감산하여 image sharpening
    frame_temp = cv.GaussianBlur(frame, (0, 0), 3)
    frame = cv.addWeighted(frame, 2.0, frame_temp, -1.0, 0)
    # frame = cv.filter2D(frame, -1, kernel)




    frame = cv.Canny(frame, 100, 200)




    '''
    frame_lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(frame_lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    frame_lab = cv.merge((cl, a, b))
    frame = cv.cvtColor(frame_lab, cv.COLOR_LAB2BGR)
    '''

    # frame의 특징점 탐색 및 저장
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    # 특징점 탐색에 대한 예외 처리
    # 특징점 데이터가 있을 경우 : 다음 절차
    #
    # 특징점 데이터가 없을 경우 : frame을 그대로 출력
    output = frame.copy()
    if des_frame is not None:
        # logo와 frame의 KNN 매칭 수행
        # 매칭 결과에 대한 예외 처리
        # 최소 매칭 횟수 이상의 횟수가 나타났을 경우 : 다음 절차
        #
        # 최소 매칭 횟수 미만의 횟수가 나타났을 경우 : frame을 그대로 출력
        #print(des_frame, "                    frame", len(frame), len(frame[0]))
        try:
            flann_match = flann.knnMatch(des_logo, des_frame, k=2)
        except:
            flann_match = []
        if len(flann_match) >= minimum_match:
            # 매칭 데이터에서 더 좋은 데이터를 선별
            # 데이터의 single element에 대한 예외 처리 수행
            good_match = []
            for p in flann_match:
                if len(p) > 1:
                    if p[0].distance < p[1].distance*ratio:
                        good_match.append(p[0])

            # 선별 결과에 대한 예외 처리
            # 최소 매칭 횟수 이상의 횟수가 나타났을 경우 : 다음 절차
            #
            # 최소 매칭 횟수 미만의 횟수가 나타났을 경우 : 마스크를 지정하지 않음
            match_mask = None
            if len(good_match) >= minimum_match:

                # logo에 대한 선별된 매칭 특징점에서 frame에 대한 선별된 매칭 특징점으로의 변환 행렬 도출
                # 행렬의 존재 여부에 따라 예외 처리
                # 존재할 경우 (대상 매칭이 가능할 경우) : 다음 절차
                #
                # 존재하지 않을 경우 (대상 매칭이 불가능할 경우) : 마스크를 지정하지 않음
                src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                if M is not None:

                    # logo의 size 좌표를 변환 행렬에 따라 변환 시킨 좌표를 도출 및 저장
                    # height, width, c = logo.shape
                    height, width = logo.shape
                    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)

                    # 해당 좌표가 frame의 밖을 벗어나는 경우에 대한 예외 처리
                    # 벗어나지 않을 경우 : 다음 절차
                    #
                    # 벗어날 경우 : 마스크를 지정하지 않음
                    able = True
                    # height, width, c = frame.shape
                    height, width = frame.shape
                    for i in dst:
                        if i[0][0] >= width or i[0][1] >= height or any(i[0] < 0):
                            able = False
                            break

                    # 해당 좌표가 내각이 모두 음각인 사각형을 이루는 지 판별, 그에 대한 예외 처리
                    # 참 : 사각형을 frame에 출력하고 해당 사각형 내의 매칭으로 마스크 설정
                    #
                    # 거짓 : 마스크를 지정하지 않음
                    if able and crossPointCheck(dst):
                        # print(dst)
                        frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                        match_mask = mask.ravel().tolist()
                        match_counter += 1
                    else:
                        match_mask = None

            output = cv.drawMatches(logo, kp_logo, frame, kp_frame, good_match,
                                    outImg=None, matchColor=None, matchesMask=match_mask, flags=2)

    frame_counter += 1
    cv.imshow("logo anc news matched", output)
    print(match_counter, '/', frame_counter, ' ', match_counter/frame_counter*100, '%')

    if cv.waitKey(5) == 27:
        break

news.release()
