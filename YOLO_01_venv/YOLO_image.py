import numpy as np
import cv2 as cv
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_source", type=str, default="./image/")
parser.add_argument("-y", "--yolo_source", type=str, default="./yolo-coco/")
parser.add_argument("-c", "--confidence", type=float, default=0.5)
parser.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(parser.parse_args())

image_path = args["image_source"]
yolo_path = args["yolo_source"]
confidence = args["confidence"]
threshold = args["threshold"]

label_path = yolo_path + "coco.names"
config_path = yolo_path + "yolov3.cfg"
weight_path = yolo_path + "yolov3.weights"

np.random.seed(42)
label = open(label_path).read().strip().split("\n")
color = np.random.randint(0, 255, size=(len(label), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv.dnn.readNetFromDarknet(config_path, weight_path)

image = cv.imread(image_path + "airplane.jpg")
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layer_output = net.forward(ln)
end = time.time()
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

boxes = []
confidences = []
class_ids = []

for output in layer_output:
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
            confidences.append((float(conf)))
            class_ids.append((class_id))

idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        c = [int(p) for p in color[class_ids[i]]]
        cv.rectangle(image, (x, y), (x+w, y+h), c, 2)
        text = "{}: {:.4f}".format(label[class_ids[i]], confidences[i])
        cv.putText(image, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

cv.imshow("image", image)
cv.waitKey(0)