
# start
print(__doc__)

import sys
try:
    video_src = sys.argv[1]
except:
    video_src = 0
App(video_src).run()

# App init
self.cap = video.create_capture(src)

# video.create_capture
source = str(source).strip()
chunks = source.split(':')

if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
    chunks[1] = chunks[0] + ':' + chunks[1]
    del chunks[0]

source = chunks[0]
try:
    source = int(source)
except ValueError:
    pass
params = dict(s.split('=') for s in chunks[1:])

cap = None
if source == 'synth':
    Class = classes.get(params.get('class', None), VideoSynthBase)
    try:
        cap = Class(**params)
    except:
        pass
else:
    cap = cv2.VideoCapture(source)
    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
if cap is None or not cap.isOpened():
    print('Warning: unable to open video source: ', source)
    if fallback is not None:
        return create_capture(fallback, None)
return cap

self.frame = None
self.paused = False
self.tracker = PlaneTracker()

cv2.namedWindow('plane')
self.rect_sel = common.RectSelector('plane', self.on_rect)

# App(video_src).run()
while True:
    playing = not self.paused and not self.rect_sel.dragging

# RectSelector init
self.win = win
self.callback = callback
cv2.setMouseCallback(win, self.onmouse)
self.drag_start = None
self.drag_rect = None