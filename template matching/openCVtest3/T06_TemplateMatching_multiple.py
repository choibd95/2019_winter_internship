import cv2 as cv
import numpy as np
import threading
import time

# threshold     : 매칭 정확도 기준, 높을수록 정확
threshold = 0.50

# wait, repeat  : 알람 쓰레드 진입 제어를 위한 세마포어
# font          : 출력 text 글꼴
wait = threading.Lock()
repeat = threading.Lock()
font = cv.FONT_HERSHEY_SIMPLEX


# 감마값 조정 함수
def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])

   return cv.LUT(image.astype(np.uint8), table.astype(np.uint8))


# 일정 시간 간격으로 방송 판별하기 위한 interval timer
# 별도 thread로 동작하며 일정 시간 동안 sleep 후 semaphore 값의 변화를 통해 signal
# main에서의 동작 방식은 synchronized
def alarm():
    repeat.acquire()

    while True:
        repeat.release()
        wait.acquire()
        time.sleep(5)
        wait.release()
        repeat.acquire()


start_time = time.perf_counter()

# logo image 로드
# height, width, scale 공간 지정
# height, width     : logo image의 높이, 너비
# scale             : 매칭에 가장 적합한 size 조정 계수
logo_name = ["SBS", "KBS", "MBC", "JTBC"]
logo = [cv.imread("T06/"+str(logo_name[i])+"_logo.png") for i in range(len(logo_name))]
logo_height, logo_width = [[0]*len(logo), [0]*len(logo)]
logo_scale = [0]*len(logo)

# 동영상 파일 로드
news = cv.VideoCapture("T06/JTBC.mp4")

# frame_counter         : 영상의 프레임 계산
# match_counter         : 각각의 logo에 대한 매칭된 프레임 계산
# min_best_matched      : 매칭되었을 경우 각각의 logo 별 매칭된 값의 상한의 최솟값 저장
# max_worst_matched     : 매칭되었을 경우 각각의 logo 별 매칭된 값의 하한의 최댓값 저장
# match_sum             : 매칭된 값 상한의 총합, 이후 평균 인식률 계산에 사용
# frame_output          : 각각의 logo 별 매칭 결과를 저장, 이후 최종 종합 결과 출력에 사용
frame_counter = 0
match_counter = [0]*len(logo)

min_best_matched = [100]*len(logo)
max_worst_matched = [0]*len(logo)
match_sum = [0]*len(logo)
frame_output = [None]*len(logo)

frame_counter_section = 0
match_counter_section = [0]*len(logo)
match_sum_section = [0]*len(logo)
match_rate_section = [0]*len(logo)
match_section = [0]*len(logo)

alarm_thread = threading.Thread(target=alarm)
alarm_thread.daemon = True

while True:
    # frame 단위로 분할
    # 에러 또는 영상의 끝일 경우에 대한 예외 처리
    ret, frame = news.read()
    if frame is None:
        print("frame is None")
        break
    cv.imshow("original frame", frame)

    # frame 내 logo가 있을 것으로 예상되는 부분을 자르고 확대
    # frame의 원본은 출력할 output을 위해
    # frame 사본은 logo와의 대조를 위해
    frame_height, frame_width = frame.shape[:2]
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

    # 영상의 첫 프레임을 읽어왔을 시 대조에 적합한 logo 크기를 구하지 못했으므로 이를 계산
    # 각각의 logo 별로 다른 size 계수를 탐색
    if logo_scale[0] == 0:
        for n in range(len(logo)):
            # 0.05부터 0.30까지 0.001 단위로 탐색
            # logo 사본을 대상으로 주어진 계수로 size 조정, 색상 영역 변경, 감마값 조정, 블러 처리
            # 대조를 수행하고 그 최댓값을 도출, 최댓값이 이전까지의 최댓값들보다 크다면 그 값과 계수를 저장
            max_temp = 0
            for size in np.linspace(0.05, 0.30, 251):
                logo_copied = cv.resize(logo[n], None, fx=size, fy=size, interpolation=cv.INTER_AREA)
                logo_copied = cv.cvtColor(logo_copied, cv.COLOR_BGR2GRAY)
                logo_copied = adjust_gamma(logo_copied, 1.3)
                logo_copied = cv.GaussianBlur(logo_copied, (0, 0), 3)
                try:
                    diff_temp = cv.matchTemplate(frame_match, logo_copied, cv.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff_temp)
                    if max_temp < max_val:
                        max_temp = max_val
                        logo_scale[n] = size
                except:
                    continue

            # 최종적으로 도출된 가장 적합한 계수로 logo 원본의 size 조정, 색상 영역 변경, 감마값 조정, 블러 처리
            logo[n] = cv.resize(logo[n], None, fx=logo_scale[n], fy=logo_scale[n], interpolation=cv.INTER_AREA)
            logo[n] = cv.cvtColor(logo[n], cv.COLOR_BGR2GRAY)
            logo[n] = adjust_gamma(logo[n], 1.3)
            logo[n] = cv.GaussianBlur(logo[n], (0, 0), 3)
            logo_height[n], logo_width[n] = logo[n].shape[:2]
            cv.imshow("logo "+str(n), logo[n])
            print("logo "+str(n)+" size ratio is decided :", logo_scale[n])

        alarm_thread.start()
        start_time = time.process_time()

    # 프레임 수 증가
    frame_counter += 1
    frame_counter_section += 1

    # 각각의 logo 별 frame과 대조
    for n in range(len(logo)):
        diff = cv.matchTemplate(frame_match, logo[n], cv.TM_CCOEFF_NORMED)
        loc = np.where(diff >= threshold)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff)
        # 최대 cost 값이 도출되는 지점을 적색 직사각형으로 출력
        # threshold 대비 적정 cost 값이 도출되는 지점을 녹색 직사각형으로 출력
        frame_output[n] = frame.copy()
        cv.rectangle(frame_output[n], max_loc, (max_loc[0] + logo_width[n], max_loc[1] + logo_height[n]), (0, 0, 255), 2)
        for point in zip(*loc[::-1]):
            cv.rectangle(frame_output[n], point, (point[0] + logo_width[n], point[1] + logo_height[n]), (0, 255, 0), 2)

        # threshold에 따른 적정 매칭이 적어도 하나 존재할 경우,
        # 매칭된 값을 종합하여 상한과 하한을 도출
        # 상한값을 합계에 더하고, 상하한이 각각 지금까지 최소, 최대인지를 확인
        # 매칭 횟수 증가
        if len(loc[0]) > 0:
            tmp_val = []
            for p in range(len(loc[0])):
                tmp_val.append(diff[loc[0][p]][loc[1][p]])

            tmp_max_val = max(tmp_val)
            tmp_min_val = min(tmp_val)
            # print("\nframe", frame_counter, ":")
            # print("logo "+str(n)+" max :", tmp_max_val)
            # print("logo "+str(n)+" min :", tmp_min_val)

            match_sum[n] += tmp_max_val
            match_sum_section[n] += tmp_max_val
            # text_num = float(int(tmp_max_val*1000)/10)

            if min_best_matched[n] > tmp_max_val:
                min_best_matched[n] = tmp_max_val
            if max_worst_matched[n] < tmp_min_val:
                max_worst_matched[n] = tmp_min_val

            match_counter[n] += 1
            match_counter_section[n] += 1

        div = match_counter_section[n] if match_counter_section[n] else 1

        text_board = np.zeros(frame.shape, np.uint8)
        cv.putText(frame_output[n], logo_name[n], (0, 25), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(text_board, str(float(int(match_sum_section[n]/div*1000)/10))+'|'+str(match_counter_section[n]),
                   (0, 25), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(text_board, str(match_section[n]), (0, 51), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        frame_output[n] = np.hstack((frame_output[n], text_board))


    # 모든 logo에 대한 output을 수직 방향으로 연장하여 출력
    output = frame_output[0].copy()
    for n in range(1, len(logo)):
        output = np.vstack((output, frame_output[n]))
    cv.imshow("output", output)

    if not wait.locked():

        for n in range(len(logo)):
            match_rate_section[n] = match_sum_section[n] / match_counter_section[n] if match_counter_section[n] else 0

        match_max_index = match_rate_section.index(max(match_rate_section))
        match_section[match_max_index] += 1
        print(sum(match_section), " try :", logo_name[match_max_index])
        print(frame_counter)

        frame_counter_section = 0
        match_counter_section = [0]*len(logo)
        match_sum_section = [0]*len(logo)

        repeat.release()

    if not repeat.locked() and wait.locked():
        repeat.acquire()

    '''
    if not frame_counter%30:
        end_time = time.process_time()
        print(frame_counter, "frame")
        print("time interval / real 1 sec :", end_time-start_time)
        start_time = time.process_time()
    '''

    if cv.waitKey(5) == 27:
        break

news.release()
cv.destroyAllWindows()

match_average = [match_sum[i]/match_counter[i] for i in range(len(logo))]

print('\n')
print("threshold :", threshold)
for i in range(len(logo)):
    print('\n')
    print(logo_name[i] + " logo")
    print("scale ratio          :", logo_scale[i])
    print("minimum best cost    :", min_best_matched[i])
    print("maximum worst cost   :", max_worst_matched[i])
    print("average max cost     :", match_average[i])
    print("matched / frame      :", match_counter[i], '/', frame_counter)
    print("match ratio          :", match_counter[i]/frame_counter*100, '%')

print('\n')
print("the mostly matched :", logo_name[match_counter.index(max(match_counter))])
print("the mostly correct :", logo_name[match_average.index(max(match_average))])

end_time = time.perf_counter() - start_time

print(end_time)