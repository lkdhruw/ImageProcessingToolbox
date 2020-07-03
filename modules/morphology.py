from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d3/dbe/tutorial_opening_closing_hats.html
    operation = cv2.MORPH_OPEN
    if 'operation' in kwargs:
        value = kwargs['operation']
        if value == 'opening':
            operation = cv2.MORPH_OPEN
        elif value == 'closing':
            operation = cv2.MORPH_CLOSE
        elif value == 'gradient':
            operation = cv2.MORPH_GRADIENT
        elif value == 'blackhat':
            operation = cv2.MORPH_BLACKHAT
        elif value == 'tophat':
            operation = cv2.MORPH_TOPHAT
        elif value == 'hitmiss':
            operation = cv2.MORPH_HITMISS

    morph_type = cv2.MORPH_ELLIPSE
    if 'type' in kwargs:
        value = kwargs['type']
        if value == 'rect':
            morph_type = cv2.MORPH_RECT
        elif value == 'cross':
            morph_type = cv2.MORPH_CROSS
        elif value == 'ellipse':
            morph_type = cv2.MORPH_ELLIPSE

    kernel_size = int(float(kwargs['ksize'])) if 'ksize' in kwargs else 3

    kernel = None
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

    if kernel is None:
        kernel = cv2.getStructuringElement(morph_type, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                           (kernel_size, kernel_size))

    morph_image = cv2.morphologyEx(image, operation, kernel)
    return morph_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Morphology',  # This name will be displayed on the menu of set True
    'function_name': 'morphology',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'operation': 'opening',  # opening | closing | gradient | tophat | blackhat | hitmiss
        'type': 'ellipse',  # rect | cross | ellipse
        'ksize': '5'  # 2n + 1
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
