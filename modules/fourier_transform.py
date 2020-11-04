from modules.features import Features
import cv2
import numpy as np


def func(image, **kwargs):
    # https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ft_image = np.fft.fft2(image)
    fshift = np.fft.fftshift(ft_image)

    '''
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum = 255 * magnitude_spectrum  # Now scale by 255
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_BGR2GRAY)
    '''
    return abs(fshift.real)/255


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'FFT',  # This name will be displayed on the menu of set True
    'function_name': 'fft',  # Optional, str, default: <name>
    'function': func,
    'parameters': {

    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
