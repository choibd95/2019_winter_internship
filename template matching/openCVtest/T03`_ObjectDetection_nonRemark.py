from __future__ import division
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

green = (0, 255, 0)


def show(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
    rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    img = cv.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):
    image = image.copy()
    image, contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv.fitEllipse(contour)
    cv.ellipse(image_with_ellipse, ellipse, green, 2, cv.LINE_AA)
    return image_with_ellipse


def find_strawberry(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    max_dimension = max(image.shape)
    scale = 700 / max_dimension
    image = cv.resize(image, None, fx=scale, fy=scale)

    image_blur = cv.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV)

    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    mask1 = cv.inRange(image_blur_hsv, min_red, max_red)

    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv.inRange(image_blur_hsv, min_red2, max_red2)

    mask = mask1 + mask2

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask_clean = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel)

    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    circled = circle_contour(overlay, big_strawberry_contour)
    show(circled)

    bgr = cv.cvtColor(circled, cv.COLOR_RGB2BGR)

    return bgr


image = cv.imread('T03/strawberry_input2.jpg')
result = find_strawberry(image)
#cv.imwrite('T03/strawberry_output.jpg', result)
cv.imshow("result", result)
cv.waitKey(0)