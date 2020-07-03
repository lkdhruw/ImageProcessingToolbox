from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/db/df6/tutorial_erosion_dilatation.html

    erosion_size = int(float(kwargs['size'])) if 'size' in kwargs else 3
    erosion_type = cv2.MORPH_ELLIPSE
    if 'type' in kwargs:
        value = kwargs['type']
        if value == 'rect':
            erosion_type = cv2.MORPH_RECT
        elif value == 'cross':
            erosion_type = cv2.MORPH_CROSS
        else:
            # value == 'ellipse'
            erosion_type = cv2.MORPH_ELLIPSE

    element = cv2.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    erosion_image = cv2.erode(image, element)
    return erosion_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Erosion',  # This name will be displayed on the menu of set True
    'function_name': 'erosion',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'size': '3',
        'type': 'ellipse'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
