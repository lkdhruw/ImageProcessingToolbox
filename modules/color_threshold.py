from modules.features import Features
import cv2
import numpy as np


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/da/d97/tutorial_threshold_inRange.html

    hsv_lower = np.array([0, 0, 0])
    hsv_higher = np.array([360, 255, 255])
    if 'hsv_lower' in kwargs:
        value = Features.decrypt_command(str(kwargs['hsv_lower'])).replace('[', '').replace(']', '')
        hsv_lower = np.array(list(map(int, value.split(','))))
    elif 'hsv_higher' in kwargs:
        value = Features.decrypt_command(str(kwargs['hsv_higher'])).replace('[', '').replace(']', '')
        hsv_higher = np.array(list(map(int, value.split(','))))

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_threshold_mask = cv2.inRange(hsv_image, hsv_lower, hsv_higher)
    contours, _ = cv2.findContours(color_threshold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return color_threshold_mask


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Color Threshold',  # This name will be displayed on the menu of set True
    'function_name': 'color_threshold',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'hsv_lower': '[0,0,0]',
        'hsv_higher': '[360,255,255]'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
