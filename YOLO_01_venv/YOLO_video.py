import numpy as np
import cv2
import imutils
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video_source", type=str, default="./video/")
parser.add_argument("-y", "--yolo_source", type=str, default="./yolo-coco/")
parser.add_argument("-c", "--confidence", type=float, default=0.5)
parser.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(parser.parse_args())

video_path = args["video_source"]
yolo_path = args["yolo_source"]
confidence = args["confidence"]
threshold = args["threshold"]

label_path = yolo_path + "coco.names"
config_path = yolo_path + "yolov3.cfg"
weight_path = yolo_path + "yolov3.weights"

np.random.seed(42)
label = open(label_path).read().strip().split("\n")
palette = np.random.randint(0, 255, size=(len(label), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weight_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("video/object sample.mp4")
writer = None
(W, H) = (None, None)

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(cap.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    frame_output = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    class_ids = []

    for output in frame_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(conf))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(p) for p in palette[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(label[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("frame", frame)
    '''
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("./output/", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

        writer.write(frame)
    '''
    if cv2.waitKey(5) == 27:
        break

if not writer is None:
    writer.release()
cap.release()