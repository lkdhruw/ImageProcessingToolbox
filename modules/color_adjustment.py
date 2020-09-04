import gc

import numpy as np

from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

    contrast = float(kwargs['contrast']) if 'contrast' in kwargs else 1.0
    brightness = float(kwargs['brightness']) if 'brightness' in kwargs else 0.0

    if 'scurve' in kwargs:
        points = Features.get_kernel(**{'data': kwargs['scurve']})
        corrected_image = image.copy()
        control_points = [[0, 0], [points[0, 0], points[0, 1]], [points[1, 0], points[1, 1]], [255, 255]]
        pts = np.array(control_points)
        point_map = {}

        def bezier(t):
            px = pts[:, 0] * np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
            py = pts[:, 1] * np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
            point_map[int(np.sum(px))] = int(np.sum(py))

        for i in range(512):
            bezier((i / 511))

        for i in range(255):
            value_flags = image == i
            corrected_image[value_flags] = point_map[i]
        gc.collect()
    elif 'zcurve' in kwargs:
        points = Features.get_kernel(**{'data': kwargs['zcurve']})
        corrected_image = image.copy()
        shadows_flags = image < int(points[0, 0])
        corrected_image[shadows_flags] = 0
        highlights_flags = image > int(points[0, 1])
        corrected_image[highlights_flags] = 255
    else:
        corrected_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    return corrected_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Color Adjustment',  # This name will be displayed on the menu of set True
    'function_name': 'color_adjustment',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'zcurve': '[30, 225]'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
