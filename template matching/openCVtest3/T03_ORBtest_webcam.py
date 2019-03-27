import cv2 as cv
import numpy as np

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

# minimumMatch  : 최소 특징점 매칭 횟수
# ratio         : 정확도 측정에 사용할 비율, 작을수록 정확
minimumMatch = 10
ratio = 0.8

# 인식할 이미지 로드
# 크기 조정, ORB를 통한 특징점 탐색 및 저장
mark = cv.imread("T03/cup_mark_small.jpg")
mark = cv.resize(mark, (320, 240), interpolation=cv.INTER_AREA)
# mark = cv.cvtColor(mark, cv.COLOR_BGR2GRAY)

mark_temp = cv.GaussianBlur(mark, (0, 0), 3)
mark = cv.addWeighted(mark, 2.0, mark_temp, -1.0, 0)

orb = cv.ORB_create()
kp_mark, des_mark = orb.detectAndCompute(mark, None)

# FLANN matcher 정의
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=200)
flann = cv.FlannBasedMatcher(index_params, search_params)

# webcam 캡처를 위한 접근 객체 선언 및 예외 처리
capture = cv.VideoCapture(0)
if capture.isOpened() == False:
    print("cam is not opened")
    exit()

while True:
    # 객체에서 촬영 영상을 읽음, 예외 처리
    ret, frame = capture.read()
    if frame is None:
        print("cam doesn't capture")
        break
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_temp = cv.GaussianBlur(frame, (0, 0), 3)
    frame = cv.addWeighted(frame, 2.0, frame_temp, -1.0, 0)

    # ORB를 통한 frame의 특징점 탐색 및 저장
    # 해당 데이터와 mark의 특징점 데이터로 FLANN matching 수행 및 저장
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    output = frame.copy()
    if des_frame is not None:

        flann_match = flann.knnMatch(des_mark, des_frame, k=2)

        if len(flann_match) >= minimumMatch:

            good_match = []
            for p in flann_match:
                if len(p) > 1:
                    if p[0].distance < p[1].distance * ratio:
                        good_match.append(p[0])

            match_mask = None
            if len(good_match) >= minimumMatch:
                src_pts = np.float32([kp_mark[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                if M is not None:
                    height, width, c = mark.shape
                    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1,
                                                                                                                 2)
                    dst = cv.perspectiveTransform(pts, M)

                    able = True
                    height, width, c = frame.shape
                    for i in dst:
                        if i[0][0] >= width or i[0][1] >= height or any(i[0] < 0):
                            able = False
                            break

                    if able and crossPointCheck(dst):
                        frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                        match_mask = mask.ravel().tolist()
                    else:
                        match_mask = None

            else:
                # print("no enough matching to find mark: ", len(goodMatch), "/", minimumMatch)
                matchMask = None

            # 매칭에 대한 결과를 시각화
            # mask가 None이 아닐 경우, 선별된 매칭 결과만이 출력되고 인식한 object를 직사각형으로 표시
            # mask가 None일 경우, 비선별 상태의 모든 매칭 결과가 출력되고 object를 표시하지 않음
            output = cv.drawMatches(mark, kp_mark, frame, kp_frame, good_match,
                                    outImg=None, matchColor=None, matchesMask=match_mask, flags=2)

    cv.imshow("mark and frame matched", output)

    if cv.waitKey(5) == 27:
        break

capture.release()
