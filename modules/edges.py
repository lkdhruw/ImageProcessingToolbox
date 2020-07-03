import re

from modules.features import Features
import cv2
import modules.smooth as sm


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d2/d2c/tutorial_sobel_derivatives.html
    # https://docs.opencv.org/4.2.0/d5/db5/tutorial_laplace_operator.html
    # https://docs.opencv.org/4.2.0/da/d5c/tutorial_canny_detector.html
    # Other operator
    # https://en.wikipedia.org/wiki/Prewitt_operator
    # https://en.wikipedia.org/wiki/Kirsch_operator

    operator = kwargs['operator'] if 'operator' in kwargs else 'canny'
    ddepth = int(float(kwargs['depth'])) if 'depth' in kwargs else -1
    ksize = int(float(kwargs['ksize'])) if 'ksize' in kwargs else 3
    scale = int(float(kwargs['scale'])) if 'scale' in kwargs else 1
    delta = 0
    border_type = cv2.BORDER_DEFAULT
    smooth = True if 'smooth' in kwargs else False
    smooth_filter = kwargs['smooth'] if smooth is True else 'gaussian3'
    ratio = int(float(kwargs['ratio'])) if 'ratio' in kwargs else 3
    threshold = int(float(kwargs['threshold'])) if 'threshold' in kwargs else 60
    threshold1 = int(float(kwargs['threshold1'])) if 'threshold1' in kwargs else threshold
    threshold2 = int(float(kwargs['threshold2'])) if 'threshold2' in kwargs else threshold*ratio

    if smooth:
        m = re.search(r'(\w+)(\d*)', smooth_filter, re.I)
        filter = m.group(1)
        filter_ksize = 3
        if not m.group(2) == '':
            filter_ksize = int(m.group(2))
        smooth_fun = sm.feature['function']
        image = smooth_fun(image, **{'ksize': filter_ksize, 'filter': filter})

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operator == 'sobel':
        grad_x = cv2.Sobel(gray_image, ddepth, 1, 0, ksize=ksize, scale=scale,
                           delta=delta, borderType=border_type)
        grad_y = cv2.Sobel(gray_image, ddepth, 0, 1, ksize=ksize, scale=scale,
                           delta=delta, borderType=border_type)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif operator == 'scharr':
        grad_x = cv2.Scharr(gray_image, ddepth, 1, 0, scale=scale,
                            delta=delta, borderType=border_type)
        grad_y = cv2.Scharr(gray_image, ddepth, 0, 1, scale=scale,
                            delta=delta, borderType=border_type)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif operator == 'laplacian':
        grad = cv2.Laplacian(gray_image, ddepth, ksize=ksize)
        edges = cv2.convertScaleAbs(grad)
    else:
        # canny operator
        im_edges = cv2.Canny(gray_image, threshold1, threshold2, ksize)
        mask = im_edges != 0
        edges = image * (mask[:, :, None].astype(image.dtype))

    return edges


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Edges',  # This name will be displayed on the menu of set True
    'function_name': 'edges',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'operator': 'canny',  # opening | closing | gradient | tophat | blackhat | hitmiss
        'ksize': '5',  # 2n + 1
        'smooth': 'gaussian3'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
