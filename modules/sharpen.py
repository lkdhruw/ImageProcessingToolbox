from modules.features import Features
import cv2
import modules.smooth as sm


def func(image, **kwargs):
    # https://www.taylorpetrick.com/blog/post/convolution-part3#sharpen
    # https://stackoverflow.com/a/32455269/2669814
    smooth_filter = sm.feature['function']
    blurr = smooth_filter(image, **kwargs)
    unsharp_image = cv2.addWeighted(image, 1.5, blurr, -0.5, 0, image)
    return unsharp_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Sharpen',  # This name will be displayed on the menu of set True
    'function_name': 'sharpen',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'ksize': '5',
        'filter': 'gaussian'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
