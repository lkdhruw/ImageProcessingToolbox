from modules.features import Features
import cv2


def func(image):
    # TODO
    pass


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Bitwise',  # This name will be displayed on the menu of set True
    'function_name': 'bitwise',  # Optional, str, default: <name>
    'function': func
}

Features.collection.append(feature)
