"""
import cv2
from modules.features import Features


def func(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


feature = {
    'name': 'Grayscale',
    'function': func
}

Features.collection.append(feature)

"""


class Features:
    collection = []
