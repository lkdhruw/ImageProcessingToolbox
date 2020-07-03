from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/db/df6/tutorial_erosion_dilatation.html
    dilatation_size = int(float(kwargs['size'])) if 'size' in kwargs else 3
    dilatation_type = cv2.MORPH_ELLIPSE

    if 'type' in kwargs:
        value = kwargs['type']
        if value == 'rect':
            dilatation_type = cv2.MORPH_RECT
        elif value == 'cross':
            dilatation_type = cv2.MORPH_CROSS
        else:
            # value == 'ellipse'
            dilatation_type = cv2.MORPH_ELLIPSE

    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_image = cv2.dilate(image, element)
    return dilatation_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Dilatation',  # This name will be displayed on the menu of set True
    'function_name': 'dilatation',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'size': '3',
        'type': 'ellipse'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
