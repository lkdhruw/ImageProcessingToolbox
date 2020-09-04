from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Grayscale',  # This name will be displayed on the menu of set True
    'function_name': 'grayscale',  # Optional, str, default: <name>
    'function': func,
    'parameters': {}  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
