import cv2 as cv
import numpy as np
import os
import threading
import time

import logo
import news
import common

font = cv.FONT_HERSHEY_SIMPLEX


class App:
    # constructor
    # logo_list             : 로고 이미지 파일에 대한 logo.Logo 객체
    # news_cast             : 뉴스 영상 파일에 대한 news.News 객체
    # frame_counter         : 방송사 판별 간격에 따른 frame 수 계산
    # whole_match_counter   : 방송사 판별 횟수 계산
    # correct_counter       : 정답 횟수 계산
    # correct               : 매칭의 정답 여부를 저장
    # size_change           : 로고 이미지 사이즈 변경 여부 저장 
    def __init__(self, path):
        self.logo_list = []
        self.news_cast = news.News()
        self.loadLogoNews(path)
        self.news_cast.castNews()

        self.frame_counter = 0
        self.whole_match_counter = 0
        self.correct_counter = 0
        self.correct = False

        self.size_change = 1

    # constructor에서 입력받은 path에 따른 파일 load
    # dir_path      : logo, news 디렉토리가 존재하는 경로
    def loadLogoNews(self, dir_path):
        # 입력값 path의 디렉토리 내에서 logo 이미지 파일 탐색 및 저장
        # 이미지 파일의 형태는 '~_logo.png'
        # 파일명의 전반부로 logo.Logo 객체 초기화
        for (path, dir, files) in os.walk(dir_path + "\\logo"):
            for file_name in files:
                extension = file_name.split(".")[-1]
                if extension == "png":
                    front = file_name.split("_")[0]
                    self.logo_list.append(logo.Logo(front))
                    print(file_name+" is added to logo_list")

        # 입력값 path의 디렉토리 내에서 영상 파일 탐색 및 저장
        # 영상 파일의 형태는 '~.mp4'
        # 파일명의 전반부로 news.News 객체에 추가
        for (path, dir, files) in os.walk(dir_path + "\\news"):
            for file_name in files:
                extension = file_name.split(".")[-1]
                if extension == "mp4":
                    front = file_name.split(".")[0]
                    self.news_cast.addNews(front)
                    print(file_name+" is added to news_cast")

        # 둘 중 하나라도 찾지 못했을 경우에 대한 예외처리
        if not len(self.logo_list) or not len(self.news_cast.news_capture):
            print("input path or target error!")
            print("there's no logo image or news file in directory path")
            exit()

    # 프로그램 기능 수행
    def run(self):
        # asyncMatch를 일정 시간마다 반복 호출하기 위한 event 변수 선언
        # asyncMatch를 별도 쓰레드로 일정 시간마다 호출
        timer = threading.Event()
        timer.set()
        threading.Timer(common.timer_timequantum, self.asyncMatch, [timer]).start()

        while True:
            # 변환 matrix와 프레임을 읽어옴
            # 에러에 대한 예외처리
            transform, frame = self.news_cast.getFrame()
            if frame is None:
                print("frame is None")
                continue

            # frame 수 가산
            self.frame_counter += 1

            # 변환 matrix를 이용하여 프레임 확대 변환
            # 대조를 위한 사본 마련
            frame = cv.warpPerspective(frame, transform,
                                       (int(frame.shape[1]/common.value_cut_ratio), int(frame.shape[0]/common.value_cut_ratio)))
            frame_match = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # 로고 이미지를 편집해야 할 경우 size 변경 (소요되는 시간 측정)
            if self.size_change:
                t_size = time.perf_counter()
                for lg in self.logo_list:
                    lg.logoRatio(frame_match)
                print("logo resized : ", time.perf_counter() - t_size)
                self.size_change = 0

            # frame 단위 매칭 결과 종합
            # 지금까지의 판별 횟수와 frame 수 추가 및 출력
            output = self.makeOutput(frame_match, frame)
            cv.putText(output, str(self.frame_counter), (frame.shape[1], 25), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(output, str(self.correct_counter)+'/'+str(self.whole_match_counter),
                       (frame.shape[1], 52), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("output", output)

            # esc 키 입력 시 종료
            # ('a', 'd' 입력 시 채널 변경)
            oper = cv.waitKey(int(1000/30))
            if oper == 27:
                break
            elif oper == 97 or oper == 100:
                self.news_cast.changeNews(oper)

        # 마지막 결과 출력 및 쓰레드 종료
        timer.clear()
        print("correct results  : ", self.correct_counter, '/', self.whole_match_counter)
        print("correct ratio    : ", self.correct_counter/self.whole_match_counter*100)

    # 일정 시간마다 방송사 판별
    # 인식률 계산 및 최대 인식률을 보인 로고를 기준으로 판별
    # 인식률 계산 요소 초기화
    def asyncMatch(self, timer, t_match=0):
        # temp_rate     : 최댓값 탐색을 위한 임시 변수
        # div           : 인식률 계산 제수, zero division 예외 처리
        temp_rate = []
        div = self.frame_counter \
            if self.frame_counter \
            else 1

        # 인식률 계산, 로고 객체별 인식률 계산 요소 초기화
        for lg in self.logo_list:
            lg.match_rate = lg.match_sum / div
            temp_rate.append(lg.match_rate)
            lg.match_frame_counter = 0
            lg.match_sum = 0
        self.frame_counter = 0
        self.size_change = 1

        # 최대 인식률 index 탐색, 이를 기준으로 방송사 판별
        self.correct = False
        match_max_index = temp_rate.index(max(temp_rate))
        if self.logo_list[match_max_index].matched:
            if self.logo_list[match_max_index].logo_name == self.news_cast.news_name[self.news_cast.news_channel]:
                self.correct_counter += 1
                self.correct = True
        self.whole_match_counter += 1
        self.logo_list[match_max_index].matched = False

        # 5초 간격으로 해당 함수 반복
        if timer.is_set():
            print("matched : ", time.perf_counter() - t_match)
            threading.Timer(common.timer_timequantum, self.asyncMatch, [timer, time.perf_counter()]).start()

    # 매칭 결과 출력 및 출력 결과 종합
    # 매칭 결과의 정확도에 따른 분류는 common.value_threshold의 값을 따름
    def makeOutput(self, frame_match, frame, threshold=common.value_threshold):
        # output : 각 로고에 대한 매칭 결과 시각화
        output = []
        for lg in self.logo_list:
            # 매칭 결과 수치화, threshold에 따른 값의 정확성 검사 및 분류, 최고 정확도의 값과 위치를 탐색
            diff = cv.matchTemplate(frame_match, lg.logo_image, cv.TM_CCOEFF_NORMED)
            loc = np.where(diff >= threshold)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff)

            # 임시 출력 matrix를 마련하고 매칭된 결과 중 최댓값에 적색 직사각형, threshold에 따른 정확한 값에 녹색 직사각형 출력
            frame_output = frame.copy()
            cv.rectangle(frame_output, max_loc, (max_loc[0] + lg.logo_shape[1], max_loc[1] + lg.logo_shape[0]),
                         (0, 0, 255), 2)
            for point in zip(*loc[::-1]):
                cv.rectangle(frame_output, point, (point[0] + lg.logo_shape[1], point[1] + lg.logo_shape[0]),
                             (0, 255, 0), 2)

            # threshold에 의해 정확하다고 판단된 위치 중 가장 높은 정확도를 보이는 위치의 cost 값을 별도 저장
            # 해당 값을 cost 총합에 가산, matching 횟수 가산, 매칭 여부 갱신
            if len(loc[0]) > 0:
                temp_val = [diff[loc[0][p]][loc[1][p]]
                            for p in range(len(loc[0]))]
                temp_max_val = max(temp_val)
                lg.match_sum += temp_max_val
                lg.match_frame_counter += 1
                lg.matched = True

            # 실시간 인식률을 계산하기 위한 제수
            # zero division 예외 처리
            div = self.frame_counter \
                if self.frame_counter \
                else 1

            # 맞는 매칭일 경우에 대한 색상 저장
            # 맞을 경우 청색, 틀릴 경우 적색
            # 로고 별 비교 출력 프레임의 우측에 공백을 마련
            # 공백에 실시간 인식률 및 맞는 매칭일 경우 이를 표시하는 문구를 출력
            # 공백과 로고 별 비교 프레임을 결합
            match_color = (255, 0, 0) \
                if self.correct \
                else (0, 0, 255)
            text_board = np.zeros(frame.shape, np.uint8)
            cv.putText(text_board,
                       str(float(int(lg.match_sum / div * 1000) / 10)) + '|' + str(lg.match_frame_counter),
                       (0, 25), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            if lg.matched:
                cv.putText(text_board, "matched", (0, 52), font, 1, match_color, 2, cv.LINE_AA)
            cv.putText(frame_output, lg.logo_name, (0, 25), font, 1, (255, 255, 255), 2, cv.LINE_AA)
            output.append(np.hstack((frame_output, text_board)))

        # 원본 프레임을 로고 별 비교 프레임과 총괄하여 결합
        text_board = np.zeros(frame.shape, np.uint8)
        output_result = cv.cvtColor(frame_match, cv.COLOR_GRAY2BGR)
        output_result = np.hstack((output_result, text_board))
        for i in output:
            output_result = np.vstack((output_result, i))

        return output_result
