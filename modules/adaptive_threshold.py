from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html
    if not Features.is_grayscale(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_type = cv2.THRESH_BINARY
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    if 'threshold_type' in kwargs:
        value = kwargs['threshold_type']
        if value == 'binary':
            thresh_type = cv2.THRESH_BINARY
        elif value == 'binary_inv':
            thresh_type = cv2.THRESH_BINARY_INV

    if 'adaptive_method' in kwargs:
        value = kwargs['adaptive_method']
        if value == 'mean':
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        elif value == 'gaussian':
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    block_size = int(float(kwargs['block_size'])) if 'block_size' in kwargs else 11
    constant = int(float(kwargs['constant'])) if 'constant' in kwargs else 2

    threshold_image = cv2.adaptiveThreshold(image, 255,
                                            adaptive_method, thresh_type,
                                            block_size, constant)
    return threshold_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Adaptive Threshold',  # This name will be displayed on the menu of set True
    'function_name': 'adaptive_threshold',  # Optional, str, default: <name>
    'function': func,
    'parameters': {  # Optional, default parameters appended at the begging
        'adaptive_method': 'gaussian',
        'block_size': '11',
        'constant': '2',
        'threshold_type': 'binary'
    }
}

Features.collection.append(feature)
