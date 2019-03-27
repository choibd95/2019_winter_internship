import app
import argparse

# 입력값 정의
# path    : logo, news 디렉토리가 존재하는 경로 (기본값 ".")
parser = argparse.ArgumentParser(description="detect logo image in news video")
parser.add_argument("path", nargs='?', type=str, default=".", help="the path of files' subdir")
args = parser.parse_args()

# App constructor -> load logo, news file
# App.run() 호출, 기능 수행
a = app.App(args.path)
a.run()
