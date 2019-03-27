import numpy as np

# value_threshold       : app.App.makeOutput의 매칭 결과의 정확도 분류에 사용
# value_gamma           : logo.Logo.adjustGamma의 감마값 변환에 사용
# value_cut_ratio       : news.News.setTransform의 프레임 비율에 사용
# value_scale_space     : logo.Logo.logoRatio의 사이즈 조정 계수의 범위에 사용
value_threshold = 0.75
value_gamma = 1.3
value_cut_ratio = 6
value_scale_space = np.linspace(0.05, 0.30, 126)

timer_timequantum = 4.99
