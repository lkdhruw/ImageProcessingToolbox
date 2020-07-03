from modules.features import Features
import cv2
import numpy as np


def func(image, **kwargs):
    # https://docs.opencv.org/2.4/modules/imgproc/doc/imgproc.html
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d
    kernel = np.ones((5, 5), dtype=np.float32)

    if 'kernel' in kwargs:
        value = kwargs['kernel']
        if type(value) is list:
            kernel = value
        else:
            if '[' in value and ']' in value:
                kernel = Features.get_kernel(**{'data': value})
            elif 'ones' in value or 'zeros' in value:
                kernel = Features.get_kernel(**{'type': value})
            else:
                kernel_ = Features.get_kernel(**{'name': value})
                kernel = kernel_  # check valid kernel

    ddepth = kwargs['depth'] if 'depth' in kwargs else -1

    filter_image = cv2.filter2D(image, ddepth, kernel)
    return filter_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Filter',  # This name will be displayed on the menu of set True
    'function_name': 'filter',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'kernel': 'ones5'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
