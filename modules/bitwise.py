from modules.features import Features
import cv2


def func(image, **kwargs):
    # https://docs.opencv.org/4.2.0/d0/d86/tutorial_py_image_arithmetics.html

    operation = kwargs['operation'] if 'operation' in kwargs else 'and'

    source1 = Features.original_image_360
    source2 = image
    if 'source1' in kwargs:
        value = kwargs['source1']
        if value == 'original':
            source1 = Features.original_image_360
        else:
            source1 = Features.get_buffer(value)
    if 'source' in kwargs or 'source2' in kwargs:
        value = kwargs['source'] if 'source' in kwargs else kwargs['source2']
        if value == 'original':
            source2 = Features.original_image_360
        else:
            source2 = Features.get_buffer(value)

    mask = Features.get_buffer(kwargs['mask']) if 'mask' in kwargs else None

    bit_image = None
    if operation == 'or':
        bit_image = cv2.bitwise_or(source1, source2, mask=mask)
    elif operation == 'not':
        bit_image = cv2.bitwise_not(source2, mask=mask)
    elif operation == 'and':
        bit_image = cv2.bitwise_and(source1, source2, mask=mask)
    elif operation == 'xor':
        bit_image = cv2.bitwise_xor(source1, source2, mask=mask)
    elif operation == 'add':
        bit_image = cv2.add(source1, source2, mask=mask)

    return bit_image


feature = {
    'menu_tree': True,  # Optional, True|False, default: True
    'name': 'Bitwise',  # This name will be displayed on the menu of set True
    'function_name': 'bitwise',  # Optional, str, default: <name>
    'function': func,
    'parameters': {
        'operation': 'and',
        'source': 'original'
    }  # Optional, default parameters appended at the begging
}

Features.collection.append(feature)
