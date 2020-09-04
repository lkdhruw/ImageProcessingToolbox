from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html

    threshold = int(kwargs['threshold']) if 'threshold' in kwargs else 150
    thresh_type = kwargs['thresh_type'] if 'thresh_type' in kwargs else 'binary'

    if thresh_type == 'binary':
        thresh_type = cv2.THRESH_BINARY
    elif thresh_type == 'binary_inv':
        thresh_type = cv2.THRESH_BINARY_INV
    elif thresh_type == 'trunc':
        thresh_type = cv2.THRESH_TRUNC
    '''elif thresh_type == 'mask':
        thresh_type = cv2.THRESH_MASK'''

    if not Features.is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(image, threshold, 255, thresh_type)
    return threshold_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Binary',  # This name will be displayed on the menu of set True
    'function_name': 'binary',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'threshold': 150
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
