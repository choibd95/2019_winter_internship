import cv2 as cv
import numpy as np


def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])

   return cv.LUT(image.astype(np.uint8), table.astype(np.uint8))


# threshold         : 매칭 cost 값의 최솟값, 클 수록 정확
# frame_counter     : frame 수 저장
# match_counter     : match 횟수 저장
# min_best_matched  : match 결과 내에서 가장 좋은 cost를 보이는 값
# max_worst_matched : match 결과 내에서 가장 나쁜 cost를 보이는 값
threshold = 0.50
frame_counter = 0
match_counter = 0
min_best_matched = 100
max_worst_matched = 0

# logo를 읽어옴
logo = cv.imread("T05/JTBC_logo_2.jpg")
# 동영상 frame을 읽어오기 위한 객체 선언
news = cv.VideoCapture("T05/JTBC.mp4")

scale = None
while True:
    # 동영상을 frame 단위로 분할
    # 영상의 끝, 또는 읽기 에러에 대한 예외처리
    ret, frame = news.read()
    if frame is None:
        print("frame is none")
        break

    # logo가 있을 것으로 예상되는 부분을 잘라냄
    # 원본은 매칭 결과 출력을 위해, gray level 데이터 사본은 매칭을 위해
    frame_height, frame_width, = frame.shape[:2]
    pts1 = np.float32([[frame_width/6*5, 0],
                       [frame_width/6*5, frame_height/6],
                       [frame_width, 0],
                       [frame_width, frame_height/6]])
    pts2 = np.float32([[0, 0],
                       [0, frame_height/6],
                       [frame_width/6, 0],
                       [frame_width/6, frame_height/6]])
    transform = cv.getPerspectiveTransform(pts1, pts2)
    frame = cv.warpPerspective(frame, transform, (int(frame_width/6), int(frame_height/6)))
    frame_match = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_match = cv.GaussianBlur(frame_match, (0, 0), 3)

    cv.imshow("frame to match", frame_match)

    # match를 위한 logo의 크기 비율을 결정
    # 동영상의 첫 frame을 대상으로 일정 간격의 불연속적 값으로부터 logo size 조정 및 cost 계산
    # cost 값의 최댓값을 저장하고 기존에 도출된 최댓값과 비교하여 그 값과 scale 값을 저장
    # 최종적으로 가장 cost가 높은 scale를 도출하면 원본 logo를 조정
    if scale is None:
        tmp_val = 0

        for tmp in np.linspace(0.05, 0.30, 251):

            logo_tmp = cv.resize(logo, None, fx=tmp, fy=tmp, interpolation=cv.INTER_AREA)
            logo_tmp = cv.cvtColor(logo_tmp, cv.COLOR_BGR2GRAY)
            logo_tmp = adjust_gamma(logo_tmp, 1.3)
            logo_tmp = cv.GaussianBlur(logo_tmp, (0, 0), 3)
            try:
                diff = cv.matchTemplate(frame_match, logo_tmp, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff)
                if tmp_val < max_val:
                    tmp_val = max_val
                    scale = tmp
            except:
                continue

        logo = cv.resize(logo, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        logo = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
        logo = adjust_gamma(logo, 1.3)
        logo = cv.GaussianBlur(logo, (0, 0), 3)
        logo_height, logo_width = logo.shape[:2]
        cv.imshow("logo", logo)
        print("logo size is decided : ", scale)

    # template matching 수행 및 주어진 threshold 값에 따른 선별
    # 그리고 최대, 최소 값도 선별
    diff = cv.matchTemplate(frame_match, logo, cv.TM_CCOEFF_NORMED)
    loc = np.where(diff >= threshold)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff)

    # 최댓값이 도출된 위치를 적색으로 표시
    # 주어진 threshold 값에 따라 적절하다고 여겨지는 위치를 녹색으로 표시
    cv.rectangle(frame, max_loc, (max_loc[0]+logo_width, max_loc[1]+logo_height), (0, 0, 255), 2)
    for point in zip(*loc[::-1]):
        cv.rectangle(frame, point, (point[0]+logo_width, point[1]+logo_height), (0, 255, 0), 2)

    frame_counter += 1

    if len(loc[0]) > 0:
        print("\nframe", frame_counter, ":")
        tmp_val = []
        for p in range(len(loc[0])):
            tmp_val.append(diff[loc[0][p]][loc[1][p]])

        tmp_max_val = max(tmp_val)
        tmp_min_val = min(tmp_val)
        print("max :", tmp_max_val)
        print("min :", tmp_min_val)
        if min_best_matched > tmp_max_val:
            min_best_matched = tmp_max_val
        if max_worst_matched < tmp_min_val:
            max_worst_matched = tmp_min_val

        match_counter += 1

    cv.imshow("matching", frame)

    if cv.waitKey(5) == 27:
        break

news.release()
print('\n')
print("threshold :", threshold)
print("logo scale ratio :", scale)
print("minimum best cost when matched :", min_best_matched)
print("maximum worst cost when matched :", max_worst_matched)
print("matched / whole frame :", match_counter, '/', frame_counter)
print("correct match ratio :", match_counter/frame_counter*100, '%')
