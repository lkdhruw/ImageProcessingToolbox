from modules.features import Features
import cv2


def func(image, **kwargs):
    # TODO
    pass


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Contours',  # This name will be displayed on the menu of set True
    'function_name': 'contours',  # Optional, str, default: <name>
    'function': func,
    'parameters': {}  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
