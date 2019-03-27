import cv2 as cv
import numpy as np
import common


class Logo:
    # constructor
    # logo_name                     : 로고의 이름을 저장
    # logo_original                 : 대조를 위한 로고의 사이즈 변경에 쓰일 원본 이미지를 저장, 이는 저장되고 갱신되지 않음
    # logo_image                    : 대조를 위해 사이즈 변경된 로고의 이미지 저장
    # logo_shape                    : 로고의 픽셀 단위 높이, 너비 저장
    # logo_scale                    : 사이즈 조정에 쓰인 계수 값을 저장
    # match_frame_counter           : 해당 로고의 매칭 횟수를 프레임 단위로 저장
    # match_sum                     : 해당 로고의 매칭 cost 값의 총합을 저장
    # match_rate                    : 해당 로고의 인식률을 저장
    # matched                       : 해당 로고가 인식되었는 지 여부를 저장
    def __init__(self, logo_name):
        self.logo_name = logo_name
        self.logo_original = cv.imread("logo/"+self.logo_name+"_logo.png")
        self.logo_image = self.logo_original.copy()
        self.logo_shape = self.logo_image.shape[:2]
        self.logo_scale = 1.00

        self.match_frame_counter = 0
        self.match_sum = 0
        self.match_rate = 0
        self.matched = False

    # 이미지에 대한 감마값 변환
    # 이미지 대조를 위한 로고 이미지의 감마값 변환에 쓰임
    # 감마값은 common.value_gamma의 영향을 받음
    def adjustGamma(self, image=None, gamma=common.value_gamma):
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)])

        if image is None:
            cv.LUT(self.logo_image.astype(np.uint8), table.astype(np.uint8))
        else:
            return cv.LUT(image.astype(np.uint8), table.astype(np.uint8))

    # 일정 간격으로 로고 이미지 대조를 위한 사이즈 조정
    # 사이즈 조정 계수의 범위는 commmon.value_scale_space의 값을 따름
    # 계수별 사이즈 조정 및 그에 따른 최대 cost 값을 탐색하고 이에 해당하는 계수를 logo_image에 적용하여 저장
    def logoRatio(self, frame_match):
        max_temp = 0
        for scale in common.value_scale_space:
            logo_copy = cv.resize(self.logo_original, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            logo_copy = cv.cvtColor(logo_copy, cv.COLOR_BGR2GRAY)
            logo_copy = self.adjustGamma(logo_copy)

            try:
                diff = cv.matchTemplate(frame_match, logo_copy, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff)
                if max_temp < max_val:
                    max_temp = max_val
                    self.logo_scale = scale
            except:
                continue

        self.logo_image = cv.resize(self.logo_original, None, fx=self.logo_scale, fy=self.logo_scale, interpolation=cv.INTER_AREA)
        self.logo_image = cv.cvtColor(self.logo_image, cv.COLOR_BGR2GRAY)
        self.adjustGamma()
        self.logo_shape = self.logo_image.shape[:2]
        cv.imshow(self.logo_name, self.logo_image)
