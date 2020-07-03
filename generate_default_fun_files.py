import os

options = ['Grayscale', 'Binary', 'Adaptive Threshold',
           'Color Threshold', 'Color Adjustment',
           'Smooth', 'Sharpen', 'Blobs', 'Filter',
           'Erosion', 'Dilatation',
           'Morphology', 'Edges', 'Convex Hull', 'Bitwise']
for op in options:
    fname = op.lower().replace(' ', '_')
    txt = """from modules.features import Features
import cv2


def func(image):
    # TODO
    pass


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': '""" + op + """',  # This name will be displayed on the menu of set True
    'function_name': '""" + fname + """',  # Optional, str, default: <name>
    'function': func
}

Features.collection.append(feature)
"""
    fname += '.py'
    mode = 'a' if os.path.exists(fname) else 'w'
    with open(fname, mode) as f:
        f.write(txt)
