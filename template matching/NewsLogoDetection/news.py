import cv2 as cv
import numpy as np
import threading
import common
import random


class News:
    # constructor
    # news_name     : 영상 파일의 이름 저장
    # news_capture  : 영상의 프레임을 읽어올 객체
    # news_channel  : 현재 읽어오는 프레임과 관련된 index를 저장
    # news_frame    : 영상의 프레임 저장
    # news_reader   : 지속적으로 프레임을 갱신할 쓰레드
    # transform     : 프레임 위치 변환 matrix 저장
    # 영상의 프레임을 읽는 쓰레드를 별도로 마련, 고정된 FPS로 프레임을 갱신
    # 로고가 있을 것으로 예상되는 부분을 잘라낼 수 있는 변환 matrix를 transform으로 저장
    def __init__(self):
        self.news_name = []
        self.news_capture = []
        self.news_channel = 0

        self.news_frame = None
        self.news_reader = threading.Thread(target=self.readNews)
        self.news_reader.daemon = True

        self.transform = None

    # parameter로 받은 name 값에 따른 news_name 및 news_capture 추가
    def addNews(self, name):
        self.news_name.append(name)
        self.news_capture.append(cv.VideoCapture("news/"+name+".mp4"))

    # 프레임을 읽어올 뉴스를 무작위로 선정 (처음 실행시만)
    # 선정된 뉴스의 프레임을 읽어오는 쓰레드 시작 및 해당 프레임 변환 matrix 저장
    def castNews(self):
        self.news_channel = random.randrange(0, len(self.news_name))
        self.news_reader.start()
        self.setTransform()

    # 'a', 'd' 값을 입력하면 news_channel이 갱신되고 프레임을 읽는 객체와 읽어오는 프레임이 변경
    # 변환 matrix 갱신
    def changeNews(self, key):
        if key == 97:
            self.news_channel = (self.news_channel + len(self.news_name) - 1) % len(self.news_name)
        elif key == 100:
            self.news_channel = (self.news_channel + len(self.news_name) + 1) % len(self.news_name)
        self.setTransform()

    # transform의 계산 및 저장
    # common.value_cut_ratio가 잘라내는 프레임의 비율에 관여
    def setTransform(self):
        ratio = common.value_cut_ratio
        frame_width = self.news_capture[self.news_channel].get(cv.CAP_PROP_FRAME_WIDTH)
        frame_height = self.news_capture[self.news_channel].get(cv.CAP_PROP_FRAME_HEIGHT)
        pts1 = np.float32([[frame_width / ratio * (ratio - 1), 0],
                           [frame_width / ratio * (ratio - 1), frame_height / ratio],
                           [frame_width, 0],
                           [frame_width, frame_height / ratio]])
        pts2 = np.float32([[0, 0],
                           [0, frame_height / ratio],
                           [frame_width / ratio, 0],
                           [frame_width / ratio, frame_height / ratio]])
        self.transform = cv.getPerspectiveTransform(pts1, pts2)

    # 영상의 프레임을 읽는 쓰레드 함수
    # 30FPS 기준으로 동작, 영상의 끝 프레임에서 처음으로 이동하고 작업 반복
    def readNews(self):
        while True:
            ret, self.news_frame = self.news_capture[self.news_channel].read()
            if self.news_frame is None:
                self.news_capture[self.news_channel].set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            cv.waitKey(int(1000 / 30))

    # 변환 matrix와 프레임 반환
    def getFrame(self):
        return self.transform, self.news_frame
