from modules.features import Features
import cv2
import numpy as np


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d0/d7a/classcv_1_1SimpleBlobDetector.html
    # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

    if not Features.is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()

    if 'minThreshold' in kwargs:
        params.minThreshold = int(float(kwargs['minThreshold']))
    if 'maxThreshold' in kwargs:
        params.maxThreshold = int(float(kwargs['maxThreshold']))
    if 'minArea' in kwargs:
        params.filterByArea = True
        params.minArea = int(float(kwargs['minArea']))
    if 'maxArea' in kwargs:
        params.filterByArea = True
        params.maxArea = int(float(kwargs['maxArea']))
    if 'minCircularity' in kwargs:
        params.filterByCircularity = True
        params.minCircularity = int(float(kwargs['minCircularity']))
    if 'maxCircularity' in kwargs:
        params.filterByCircularity = True
        params.maxCircularity = int(float(kwargs['maxCircularity']))
    if 'minConvexity' in kwargs:
        params.filterByConvexity = True
        params.minConvexity = int(float(kwargs['minConvexity']))
    if 'maxConvexity' in kwargs:
        params.filterByConvexity = True
        params.maxConvexity = int(float(kwargs['maxConvexity']))
    if 'minInertia' in kwargs:
        params.filterByInertia = True
        params.minInertiaRatio = int(float(kwargs['minInertia']))
    if 'maxInertia' in kwargs:
        params.filterByInertia = True
        params.maxInertiaRatio = int(float(kwargs['maxInertia']))
    if 'color' in kwargs:
        params.filterByColor = True
        params.blobColor = Features.str_to_tuple(kwargs['color'])

    draw_color = (Features.str_to_tuple(kwargs['draw_color'])) if 'draw_color' in kwargs else (0, 255, 0)

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, draw_color,
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blobs


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Blobs',  # This name will be displayed on the menu of set True
    'function_name': 'blobs',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'minThreshold': 20,
        'maxThreshold': 180,
        'minArea': '100',
        'minCircularity': '0.9'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
