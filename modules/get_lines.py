import numpy as np

from modules.features import Features
import cv2


def func(image, **kwargs):
    #  documentation

    algorithm = 'probabilistic'
    rho = 1
    theta = np.pi / 180
    threshold = 150
    lines = None
    srn = 0
    stn = 0
    min_theta = None
    max_theta = None
    minLineLength = 50
    maxLineGap = 10
    for key, value in kwargs.items():
        if 'probabilistic' in key:
            algorithm = value

    edges = cv2.Canny(image, 50, 200, None, 3)
    # edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    c_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if algorithm == 'standard':
        lines = cv2.HoughLines(edges, rho, theta, threshold, lines, srn, stn, min_theta, max_theta)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(c_edges, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        linesP = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, lines=lines,
                                 minLineLength=minLineLength, maxLineGap=maxLineGap)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(c_edges, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    lined_image = c_edges
    return lined_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Get Lines',  # This name will be displayed on the menu of set True
    'function_name': 'get_lines',  # Optional, str, default: <name>
    'function': func,
    'parameters': {}  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
