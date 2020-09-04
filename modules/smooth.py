from modules.features import Features
import cv2


def str_to_point(pt: str):
    x, y = pt.split(',')
    return int(x), int(y)

def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d4/d13/tutorial_py_filtering.html
    # Need implement more filters such as gaussian, median etc
    ksize = int(kwargs['ksize']) if 'ksize' in kwargs else 3
    ksizex = int(kwargs['ksizex']) if 'ksizex' in kwargs else ksize
    ksizey = int(kwargs['ksizey']) if 'ksizey' in kwargs else ksizex

    sigma_x = int(kwargs['sigma_x']) if 'sigma_x' in kwargs else 0
    sigma_y = int(kwargs['sigma_y']) if 'sigma_y' in kwargs else sigma_x

    diameter = int(kwargs['diameter']) if 'diameter' in kwargs else 0
    sigma_color = int(kwargs['sigma_color']) if 'sigma_color' in kwargs else 0
    sigma_space = int(kwargs['sigma_space']) if 'sigma_space' in kwargs else 0
    filter_type = kwargs['filter'] if 'filter' in kwargs else 'gaussian'

    anchor_point = (str_to_point(kwargs['anchor_point'])) if 'anchor_point' in kwargs else (-1, -1)

    if filter_type == 'box':
        blur_image = cv2.blur(image, (ksizex, ksizey))
    elif filter_type == 'median':
        blur_image = cv2.medianBlur(image, ksize)
    elif filter_type == 'bilateral':
        blur_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    else:
        # gaussian
        blur_image = cv2.GaussianBlur(image, (ksizex, ksizey), sigma_x, sigma_y)
    return blur_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Smooth',  # This name will be displayed on the menu of set True
    'function_name': 'smooth',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'ksize': '5',
        'filter': 'gaussian'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
