import os
'''
options = ['Grayscale', 'Binary', 'Adaptive Threshold',
           'Color Threshold', 'Color Adjustment',
           'Smooth', 'Sharpen', 'Blobs', 'Filter',
           'Erosion', 'Dilatation',
           'Morphology', 'Edges', 'Convex Hull', 'Bitwise']
'''
options = ['Get Lines', 'Contour']
for op in options:
    fname = op.lower().replace(' ', '_')
    txt = """from modules.features import Features
import cv2


def func(image, **kwargs):
    # TODO
    pass


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': '""" + op + """',  # This name will be displayed on the menu of set True
    'function_name': '""" + fname + """',  # Optional, str, default: <name>
    'function': func,
    'parameters': {}  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
"""
    fname += '.py'
    mode = 'a' if os.path.exists(fname) else 'w'
    with open(fname, mode) as f:
        f.write(txt)
