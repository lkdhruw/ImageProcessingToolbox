import numpy as np

from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d7/d1d/tutorial_hull.html
    threshold = 40
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_output = cv2.Canny(gray_image, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (0, 256, 0)
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    return drawing


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Convex Hull',  # This name will be displayed on the menu of set True
    'function_name': 'convex_hull',  # Optional, str, default: <name>
    'function': func,
    'parameters': {}  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
