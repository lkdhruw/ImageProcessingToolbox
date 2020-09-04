import numpy as np

from modules.features import Features
from modules.binary import func as binary
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff

    threshold = int(kwargs['threshold']) if 'threshold' in kwargs else 50
    threshold1 = int(kwargs['threshold1']) if 'threshold1' in kwargs else threshold
    threshold2 = int(kwargs['threshold2']) if 'threshold2' in kwargs else np.clip(3*threshold1, 0, 255)

    if 'mask' in kwargs:
        mask = Features.get_buffer(kwargs['mask'])
    else:
        # mask = binary(image, **kwargs)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(gray, threshold1, threshold2, apertureSize=5, L2gradient=True)

    mode = cv2.RETR_LIST
    if 'mode' in kwargs:
        value = kwargs['mode']
        if value == 'ccomp':
            mode = cv2.RETR_CCOMP
        elif value == 'external':
            mode = cv2.RETR_EXTERNAL
        elif value == 'floodfill':
            mode = cv2.RETR_FLOODFILL
        elif value == 'tree':
            mode = cv2.RETR_TREE
        else:
            mode = cv2.RETR_LIST

    method = cv2.CHAIN_APPROX_SIMPLE
    if 'method' in kwargs:
        value = kwargs['method']
        if value == 'none':
            method = cv2.CHAIN_APPROX_NONE
        elif value == 'tc89k':
            method = cv2.CHAIN_APPROX_TC89_KCOS
        elif value == 'tc89l':
            method = cv2.CHAIN_APPROX_TC89_L1

    contours, hierarchy = cv2.findContours(mask, mode, method)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image
    pass
    cnt = contours[0]
    # Moment
    moment = cv2.moments(cnt)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # Contour Approximation
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        #  Bounding Rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        kwargs['shape'] = 'line'
    # Fit rotated rectangle, circle, ellipse, and line
    shape = kwargs['shape'] if 'shape' in kwargs else 'circle'
    if shape == 'circle':
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0, 255, 0), 2)
    elif shape == 'ellipse':
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    elif shape == 'rectangle':
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    elif shape == 'line':
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    return image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Contours',  # This name will be displayed on the menu of set True
    'function_name': 'contours',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'threshold': 50,
        'shape': 'circle'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
