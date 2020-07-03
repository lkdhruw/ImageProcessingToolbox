import re
import cv2
import numpy as np


class Features:
    buffer = {}
    original_image_360 = None
    modified_image_360 = None
    collection = []

    @staticmethod
    def is_grayscale(image):
        return len(image.shape) < 3

    @staticmethod
    def str_to_tuple(string: str, dtype='int'):
        string = string.replace('[', '').replace(']', '')
        if dtype == 'string':
            nparray = np.array(string.split(','), dtype=np.str)
        elif dtype == 'float':
            nparray = np.array(string.split(','), dtype=np.float)
        else:
            nparray = np.array(string.split(','), dtype=np.int)
        return tuple(nparray)

    @staticmethod
    def encrypt_command(data: str):
        while re.search(r'\[([^\[\]]*)(,)([^\[\]]*)\]', data):
            data = re.sub(r'\[([^\[\]]*)(,)([^\[\]]*)\]', r'[\1%44%\3]', data)
        return data

    @staticmethod
    def decrypt_command(data: str):
        data = data.replace('%44%', ',')
        return data

    @staticmethod
    def get_buffer(name: str):
        return Features.buffer[name]

    @staticmethod
    def set_buffer(name: str, data):
        Features.buffer[name] = data

    @staticmethod
    def clear_buffer(self):
        Features.buffer = {}

    @staticmethod
    def get_kernel(self, **kwargs):
        kernel = None

        if 'name' in kwargs or 'type' in kwargs:
            value = kwargs['type'] if 'type' in kwargs else kwargs['name']
            if 'zeros' in value:
                value = str(value).replace('zeros', '')
                size = int(value) if value != '' else 3
                kernel = np.zeros((size, size), dtype=np.float32)
            elif 'ones' in value:
                value = str(value).replace('ones', '')
                size = int(value) if value != '' else 3
                kernel = np.ones((size, size), dtype=np.float32)
                kernel /= (size * size)
            elif 'gaussian' in value:
                value = str(value).replace('gaussian', '')
                value = value.split('_')
                size = int(value[0]) if value != '' else 3
                sigma = float(value[1]) if len(value) == 2 else 0.0
                kernel = cv2.getGaussianKernel(size, sigma)

        if 'data' in kwargs:
            data = str(kwargs['data']).replace('%44%', ',').replace('[', '').replace(']', '')
            rows = data.split(';')
            kernel = []
            for row in rows:
                row_data = row.split(',')
                kernel.append(row_data)
            kernel = np.array(kernel, dtype=np.float32)

        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.float32)/(3*3)
        return kernel
