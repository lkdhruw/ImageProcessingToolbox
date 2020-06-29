import threading
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
from PIL import ImageFilter
from PIL import ImageTk
import colorsys
import io
import re
import cv2
import binascii
import numpy as np
import colorsys
import os
import gc


class Binary:
    def __init__(self, value: int):
        self.value = value

    def command(self):
        return 'binary th=' + str(self.value) + ''


class Window:
    def __init__(self, master):
        master.title("Digital Image Processing")
        master.minsize(1200, 600)
        master.resizable(0, 0)
        self.master = master
        self.filename = ""
        self.commandText = ""

        # toolbar = Frame(root)
        # toolbar.grid(row=1, column=1)

        toolbar = LabelFrame(root)
        toolbar.grid(row=1, column=1)
        self.browseButton = Button(toolbar, text="Browse", command=self.browseimg)
        self.browseButton.grid(row=1, column=1, padx=2, pady=1)
        self.tasks = StringVar(root)
        self.tasks.set('Grayscale')
        tasks = OptionMenu(toolbar, self.tasks,
                           'Grayscale', 'Binary', 'Adaptive Threshold',
                           'Color Threshold', 'Color Adjustment'
                           'Smooth', 'Sharpen', 'Blobs', 'Filter',
                           'Erosion', 'Dilatation',
                           'Morphology', 'Edges', 'Convex Hull', 'Bitwise'
                           )
        tasks.grid(row=1, column=2, padx=2, pady=1)
        tasks.configure(width=15)
        self.addButton = Button(toolbar, text="Add", command=self.add_task)
        self.addButton.grid(row=1, column=3, padx=2, pady=1)
        self.chooseColorButton = Button(toolbar, text="Choose Color", command=self.choose_color)
        self.chooseColorButton.grid(row=1, column=4, padx=2, pady=1)
        self.camButton = Button(toolbar, text="Cam", command=self.webcam)
        self.camButton.grid(row=1, column=5, padx=2, pady=1)
        self.saveButton = Button(toolbar, text="Save", command=self.save_buffer)
        self.saveButton.grid(row=1, column=6, padx=2, pady=1)
        self.showButton = Button(toolbar, text="Show", command=self.show_buffer)
        self.showButton.grid(row=1, column=7, padx=2, pady=1)

        # im = Image.open("Default.png")
        im = cv2.imread('Default.png')
        self.originalImage = im
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        # self.originalImage480x480 = cv2.resize(im, (480, 480), interpolation=cv2.INTER_AREA)
        self.originalImage360x360 = cv2.resize(im, (360, 360), interpolation=cv2.INTER_AREA)

        self.buffers = {}
        self.cam = None  # cv2.VideoCapture(0)
        self.cam_capturing = False
        self.video = None
        self.video_playing = False

        image = self.originalImage360x360
        self.last_modified_image = image
        self.buffer_image = image
        # photo = self.imageToBytes(image)
        photo = self.arrayToImage(image)  # openCV to PIL image
        self.originalImageView = Label(root, width=400, height=360)
        self.originalImageView.grid(row=2, column=1)
        self.originalImageView["image"] = photo
        self.originalPhoto = photo
        Label(root, text="Original image").grid(row=3, column=1)

        self.modifiedImageView = Label(root, width=400, height=360)
        self.modifiedImageView.grid(row=2, column=2)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        Label(root, text="Modified image").grid(row=3, column=2)

        def command_edit(event):
            # print(repr(event.keysym))
            commands = self.command.get('1.0', 'end-1c')
            commands = str(commands).split('\r\n')
            for command in commands:
                print(command, end=', ')

        self.command = Text(root, height=10, width=80)
        self.command.grid(row=4, column=1, columnspan=2, sticky="nsew", pady=(10, 10), padx=(5, 0))
        # self.command.bind('<Key>', command_edit)

        self.commandScrollbar = Scrollbar(root, command=self.command.yview)
        self.commandScrollbar.grid(row=4, column=3, sticky="nsew", pady=(10, 10))
        self.command['yscrollcommand'] = self.commandScrollbar.set

        self.applyButton = Button(root, text="Apply", command=self.applycommand)
        self.applyButton.grid(row=5, column=2)

        self.optionsPanelCanvas = Canvas(root, width=300)
        self.optionsPanelCanvas.grid(row=1, column=4, rowspan=4, sticky="news")
        self.optionScrollbar = Scrollbar(root, command=self.optionsPanelCanvas.yview)
        self.optionScrollbar.grid(row=1, column=5, rowspan=4, sticky="ns")
        self.optionsPanelCanvas['yscrollcommand'] = self.optionScrollbar.set

        self.optionsPanelCanvas.xview_moveto(0)
        self.optionsPanelCanvas.yview_moveto(0)
        self.optionsPanel = optionsPanel = Frame(self.optionsPanelCanvas)
        self.panelID = self.optionsPanelCanvas.create_window((0, 0), window=optionsPanel, anchor='nw')

        def _configure_frame(event):
            # update the scrollbars to match the size of the inner frame
            size = (self.optionsPanel.winfo_reqwidth(), self.optionsPanel.winfo_reqheight())
            self.optionsPanelCanvas.config(scrollregion="0 0 %s %s" % size)
            if self.optionsPanel.winfo_reqwidth() != self.optionsPanelCanvas.winfo_width():
                # update the canvas's width to fit the inner frame
                self.optionsPanelCanvas.config(width=self.optionsPanel.winfo_reqwidth())

        self.optionsPanel.bind('<Configure>', _configure_frame)

        def _configure_canvas(event):
            if self.optionsPanel.winfo_reqwidth() != self.optionsPanelCanvas.winfo_width():
                # update the inner frame's width to fill the canvas
                self.optionsPanelCanvas.itemconfigure(self.panelID, width=self.optionsPanelCanvas.winfo_width())

        self.optionsPanelCanvas.bind('<Configure>', _configure_canvas)

        self.hsv_range_filter = LabelFrame(optionsPanel, text='HSV range filter', padx=1, pady=1)
        self.hsv_range_filter.grid(row=2, column=1)
        self.hsv_low = LabelFrame(self.hsv_range_filter, text='HSV low', padx=5, pady=5)
        self.hsv_low.grid(row=1, column=1)
        self.h_low = Scale(self.hsv_low, label="H", from_=0, to=360, length=300,
                           orient=HORIZONTAL)
        self.h_low.grid(row=1, column=1)
        self.h_low.bind("<B1-Motion>",
                        lambda event, obj=self.h_low, name='hsv_h_low': self.scale_move(event, obj, name))
        self.s_low = Scale(self.hsv_low, label="S", from_=0, to=255, length=300,
                           orient=HORIZONTAL)
        self.s_low.grid(row=2, column=1)
        self.s_low.bind("<B1-Motion>",
                        lambda event, obj=self.s_low, name='hsv_s_low': self.scale_move(event, obj, name))
        self.v_low = Scale(self.hsv_low, label="V", from_=0, to=255, length=300,
                           orient=HORIZONTAL)
        self.v_low.grid(row=3, column=1)
        self.v_low.bind("<B1-Motion>",
                        lambda event, obj=self.v_low, name='hsv_v_low': self.scale_move(event, obj, name))

        self.hsv_high = LabelFrame(self.hsv_range_filter, text='HSV high', padx=5, pady=5)
        self.hsv_high.grid(row=2, column=1)
        self.h_high = Scale(self.hsv_high, label="H", from_=0, to=360, length=300,
                            orient=HORIZONTAL)
        self.h_high.grid(row=1, column=1)
        self.h_high.bind("<B1-Motion>",
                         lambda event, obj=self.h_high, name='hsv_h_high': self.scale_move(event, obj, name))
        self.s_high = Scale(self.hsv_high, label="S", from_=0, to=255, length=300,
                            orient=HORIZONTAL)
        self.s_high.grid(row=2, column=1)
        self.s_high.bind("<B1-Motion>",
                         lambda event, obj=self.s_high, name='hsv_s_high': self.scale_move(event, obj, name))
        self.v_high = Scale(self.hsv_high, label="V", from_=0, to=255, length=300,
                            orient=HORIZONTAL)
        self.v_high.grid(row=3, column=1)
        self.v_high.bind("<B1-Motion>",
                         lambda event, obj=self.v_high, name='hsv_v_high': self.scale_move(event, obj, name))

    def rgb_to_hsv(self, r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h, s, v = str(int(h * 360)), str(int(s * 255)), str(int(v * 255))
        return ', '.join([h, s, v])

    def choose_color(self):
        color = askcolor()
        r, g, b = color[0]
        hsv = self.rgb_to_hsv(r, g, b)
        self.master.clipboard_clear()
        self.master.clipboard_append(hsv)

    def add_task(self):
        task = self.tasks.get().replace(' ', '_')
        append_at = self.command.index('end-1c')
        if task == 'Grayscale':
            self.command.insert(append_at, task.lower() + '\n')
        elif task == 'Binary':
            self.command.insert(append_at, task.lower() + '(128)\n')
        elif task == 'Adaptive_Threshold':
            defaults = [
                'adaptive_method=gaussian',
                'block_size=11',
                'constant=2',
                'threshold_type=binary'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Color_Threshold':
            defaults = [
                'hsv_lower=[]',
                'hsv_higher=[]'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Smooth':
            defaults = [
                'ksize=5',
                'filter=gaussian'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Sharpen':
            defaults = [
                'ksize=5',
                'filter=gaussian'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Blobs':
            defaults = [
                'minArea=100',
                'minCircularity=0.9'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Filter':
            defaults = [
                'kernel=ones5'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Erosion':
            defaults = [
                'size=3',
                'type=cross'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Dilatation':
            defaults = [
                'size=3',
                'type=cross'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Morphology':
            defaults = [
                'operation=opening',  # opening | closing | gradient | tophat | blackhat | hitmiss
                'type=cross',  # rect | cross | ellipse
                'ksize=5'  # 2n + 1
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Edges':
            defaults = [
                'operator=canny',  # opening | closing | gradient | tophat | blackhat | hitmiss
                'ksize=5',  # 2n + 1
                'smooth=gaussian3'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Convex_Hull':
            defaults = [

            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')
        elif task == 'Bitwise':
            defaults = [
                'operation=and'
            ]
            self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')

    def save_buffer(self):
        self.buffer_image = self.last_modified_image

    def show_buffer(self):
        cv2.imshow('Buffer image', self.buffer_image)

    def is_grayscale(self, image):
        return len(image.shape) < 3

    def grayscale(self, image):
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''photo = self.arrayToImage(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo'''
        return image

    def quantize(self, image, quantizeValue):
        # bug: the whole programme is now using
        # an opencv image not PIL image object
        # Need to resolve
        image = image.quantize(quantizeValue)
        return image

    def binary(self, image, threshold, thresh_type=cv2.THRESH_BINARY):
        # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image, threshold, 255, thresh_type)
        return threshold_image

    def adaptive_threshold(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_type = cv2.THRESH_BINARY
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        block_size = 11
        constant = 2
        # keys:
        #   'adaptive_method=gaussian',
        #   'block_size=11',
        #   'constant=2',
        #   'threshold_type=binary'
        for key, value in kwargs.items():
            if key == 'adaptive_method':
                if value == 'mean':
                    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
                elif value == 'gaussian':
                    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            elif key == 'block_size':
                block_size = int(float(value))
            elif key == 'constant':
                constant = int(float(value))
            elif key == 'threshold_type':
                if value == 'binary':
                    thresh_type = cv2.THRESH_BINARY
                elif value == 'binary_inv':
                    thresh_type = cv2.THRESH_BINARY_INV

        threshold_image = cv2.adaptiveThreshold(image, 255,
                                                adaptive_method, thresh_type,
                                                block_size, constant)
        return threshold_image

    def smooth(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d4/d13/tutorial_py_filtering.html
        # Need implement more filters such as gaussian, median etc
        ksize = 5
        ksizex, ksizey = ksize, ksize
        sigma_x, sigma_y = 0, 0
        diameter, sigma_color, sigma_space = 0, 0, 0
        filter_type = 'box'
        anchor_point = '-1,-1'
        for key, value in kwargs.items():
            if 'ksize' in key:
                ksize = int(value)
                ksizex, ksizey = ksize, ksize
            elif 'ksizex' in key:
                ksizex = int(value)
                ksize = ksizex
            elif 'ksizey' in key:
                ksizey = int(value)
            elif 'sigma_x' in key:
                sigma_x = float(value)
                sigma_y = sigma_x
            elif 'sigma_y' in key:
                sigma_y = float(value)
            elif 'diameter' in key:
                diameter = float(value)
            elif 'sigma_color' in key:
                sigma_color = float(value)
            elif 'sigma_space' in key:
                sigma_space = float(value)
            elif 'filter' in key:
                filter_type = value
            elif 'anchor_point' in key:
                x, y = str(value).split(',')
                anchor_point = (int(x), int(y))

        if filter_type == 'gaussian':
            blur_image = cv2.GaussianBlur(image, (ksizex, ksizey), sigma_x, sigma_y)
        elif filter_type == 'median':
            blur_image = cv2.medianBlur(image, ksize)
        elif filter_type == 'bilateral':
            blur_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        else:
            blur_image = cv2.blur(image, (ksizex, ksizey))
        return blur_image

    def blobs(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d0/d7a/classcv_1_1SimpleBlobDetector.html
        # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        draw_color = (0, 255, 0)
        params = cv2.SimpleBlobDetector_Params()
        for key, value in kwargs.items():
            if 'minArea' in key:
                params.filterByArea = True
                params.minArea = int(float(value))
            elif 'maxArea' in key:
                params.filterByArea = True
                params.maxArea = int(float(value))
            elif 'minCircularity' in key:
                params.filterByCircularity = True
                params.minArea = int(float(value))
            elif 'maxCircularity' in key:
                params.filterByCircularity = True
                params.maxArea = int(float(value))
            elif 'minConvexity' in key:
                params.filterByConvexity = True
                params.maxArea = int(float(value))
            elif 'maxConvexity' in key:
                params.filterByConvexity = True
                params.maxArea = int(float(value))
            elif 'minInertia' in key:
                params.filterByInertia = True
                params.maxArea = int(float(value))
            elif 'maxInertia' in key:
                params.filterByInertia = True
                params.maxArea = int(float(value))
            elif 'color' in key:
                params.filterByColor = True
                params.blobColor = int(float(value))
            elif 'draw_color' in key:
                draw_color = value

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(image, keypoints, blank, draw_color,
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return blobs

    def filter(self, image, **kwargs):
        # https://docs.opencv.org/2.4/modules/imgproc/doc/imgproc.html
        # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d
        kernel = np.ones((5, 5), dtype=np.float32)
        ddepth = -1
        for key, value in kwargs.items():
            if 'kernel' in key:
                if type(value) is list:
                    kernel = value
                else:
                    if '[' in value and ']' in value:
                        kernel = self.get_kernel(**{'data': value})
                    elif 'ones' in value or 'zeros' in value:
                        kernel = self.get_kernel(**{'type': value})
                    else:
                        kernel_ = self.get_kernel(**{'name': value})
                        kernel = kernel_  # check valid kernel
            elif 'depth' in key:
                ddepth = int(value)

        filter_image = cv2.filter2D(image, ddepth, kernel)
        return filter_image

    def erosion(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/db/df6/tutorial_erosion_dilatation.html
        erosion_size = 3
        erosion_type = cv2.MORPH_CROSS
        for key, value in kwargs.items():
            if 'size' in key:
                erosion_size = int(float(value))
            if 'type' in key:
                if value == 'rect':
                    erosion_type = cv2.MORPH_RECT
                elif value == 'cross':
                    erosion_type = cv2.MORPH_CROSS
                elif value == 'ellipse':
                    erosion_type = cv2.MORPH_ELLIPSE

        element = cv2.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        erosion_image = cv2.erode(image, element)
        return erosion_image

    def dilatation(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/db/df6/tutorial_erosion_dilatation.html
        dilatation_size = 3
        dilatation_type = cv2.MORPH_CROSS
        for key, value in kwargs.items():
            if 'size' in key:
                dilatation_size = int(float(value))
            elif 'type' in key:
                if value == 'rect':
                    dilatation_type = cv2.MORPH_RECT
                elif value == 'cross':
                    dilatation_type = cv2.MORPH_CROSS
                elif value == 'ellipse':
                    dilatation_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
        dilatation_image = cv2.dilate(image, element)
        return dilatation_image

    def color_threshold(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/da/d97/tutorial_threshold_inRange.html

        hsv_lower = np.array([0, 0, 0])
        hsv_higher = np.array([0, 0, 0])
        for key, value in kwargs.items():
            if 'hsv_lower' in key:
                value = self.decrypt_command(str(value)).replace('[', '').replace(']', '')
                hsv_lower = np.array(list(map(int, value.split(','))))
            elif 'hsv_higher' in key:
                value = self.decrypt_command(str(value)).replace('[', '').replace(']', '')
                hsv_higher = np.array(list(map(int, value.split(','))))

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_threshold_mask = cv2.inRange(hsv_image, hsv_lower, hsv_higher)
        contours, _ = cv2.findContours(color_threshold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        return color_threshold_mask

    def morphology(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d3/dbe/tutorial_opening_closing_hats.html
        operation = cv2.MORPH_OPEN
        morph_type = cv2.MORPH_CROSS
        kernel_size = 5
        kernel = None
        for key, value in kwargs.items():
            if 'operation' in key:
                if value == 'opening':
                    operation = cv2.MORPH_OPEN
                elif value == 'closing':
                    operation = cv2.MORPH_CLOSE
                elif value == 'gradient':
                    operation = cv2.MORPH_GRADIENT
                elif value == 'blackhat':
                    operation = cv2.MORPH_BLACKHAT
                elif value == 'tophat':
                    operation = cv2.MORPH_TOPHAT
                elif value == 'hitmiss':
                    operation = cv2.MORPH_HITMISS
            elif 'type' in key:
                if value == 'rect':
                    morph_type = cv2.MORPH_RECT
                elif value == 'cross':
                    morph_type = cv2.MORPH_CROSS
                elif value == 'ellipse':
                    morph_type = cv2.MORPH_ELLIPSE
            elif 'ksize' in key:
                kernel_size = int(float(value))
            elif 'kernel' in key:
                if type(value) is list:
                    kernel = value
                else:
                    if '[' in value and ']' in value:
                        kernel = self.get_kernel(**{'data': value})
                    elif 'ones' in value or 'zeros' in value:
                        kernel = self.get_kernel(**{'type': value})
                    else:
                        kernel_ = self.get_kernel(**{'name': value})
                        kernel = kernel_  # check valid kernel
        if kernel is None:
            kernel = cv2.getStructuringElement(morph_type, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                               (kernel_size, kernel_size))

        morph_image = cv2.morphologyEx(image, operation, kernel)
        return morph_image

    def draw_edges(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d2/d2c/tutorial_sobel_derivatives.html
        # https://docs.opencv.org/4.2.0/d5/db5/tutorial_laplace_operator.html
        # https://docs.opencv.org/4.2.0/da/d5c/tutorial_canny_detector.html
        # Other operator
        # https://en.wikipedia.org/wiki/Prewitt_operator
        # https://en.wikipedia.org/wiki/Kirsch_operator
        operator = 'canny'
        ddepth = -1
        ksize = 3
        scale = 1
        delta = 0
        border_type = cv2.BORDER_DEFAULT
        smooth = False
        smooth_filter = 'gaussian3'
        ratio = 3
        threshold = 60
        threshold1, threshold2 = threshold, threshold * ratio

        for key, value in kwargs.items():
            if 'operator' in key:
                operator = value
            elif 'ksize' in key:
                ksize = int(value)
            elif 'smooth' in key:
                smooth = True
                smooth_filter = value
            elif 'scale' in key:
                scale = int(value)
            elif 'threshold' in key:
                ratio: int = int(kwargs['ratio']) or ratio
                threshold = int(value)
                threshold1, threshold2 = threshold, threshold * ratio
            elif 'threshold1' in key:
                threshold1 = int(value)
                threshold2 = int(kwargs['threshold2'])
            elif 'ratio' in key:
                threshold1 = int(kwargs['threshold1'])
                threshold2 = threshold1 * int(value)

        if smooth:
            m = re.search(r'(\w+)(\d*)', smooth_filter, re.I)
            filter = m.group(1)
            filter_ksize = 3
            if not m.group(2) == '':
                filter_ksize = int(m.group(2))
            image = self.smooth(image, **{'ksize': filter_ksize, 'filter': filter})

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

    def draw_convex_hull(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d7/d1d/tutorial_hull.html
        threshold = 40
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny_output = cv2.Canny(gray_image, threshold, threshold * 2)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (0, 256, 0)
            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color)

        return drawing

    def bitwise(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d0/d86/tutorial_py_image_arithmetics.html
        operation = 'and'
        im = image
        mask = None
        for key, value in kwargs.items():
            if 'operation' in key:
                operation = value
            elif 'source' in key:
                if value == 'original':
                    im = self.originalImage360x360
                else:
                    im = self.get_buffer(value)
            elif 'mask' in key:
                if value == 'buffer':
                    mask = self.get_kernel(**{'name': value})
                else:
                    mask = self.get_buffer(value)
        original = self.originalImage360x360
        bit_image = None
        if operation == 'or':
            bit_image = cv2.bitwise_or(original, im, mask=mask)
        elif operation == 'not':
            bit_image = cv2.bitwise_not(im, mask=mask)
        elif operation == 'and':
            bit_image = cv2.bitwise_and(original, im, mask=mask)
        elif operation == 'xor':
            bit_image = cv2.bitwise_xor(original, im, mask=mask)
        elif operation == 'add':
            bit_image = cv2.add(image, im, mask=mask)

        return bit_image

    def sharpen(self, image, **kwargs):
        # https://www.taylorpetrick.com/blog/post/convolution-part3#sharpen
        # https://stackoverflow.com/a/32455269/2669814
        blurr = self.smooth(image, **kwargs)
        unsharp_image = cv2.addWeighted(image, 1.5, blurr, -0.5, 0, image)
        return unsharp_image

    def color_adjustment(self, image, **kwargs):
        # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        contrast = 1.0
        brightness = 0.0
        for key, value in kwargs.items():
            if 'contrast' in key:
                contrast = float(value)
            elif 'brightness' in key:
                brightness = float(value)

        if 'scurve' in kwargs:
            points = self.get_kernel(**{'data': kwargs['scurve']})
            corrected_image = image.copy()
            control_points = [[0, 0], [points[0, 0], points[0, 1]], [points[1, 0], points[1, 1]], [255, 255]]
            pts = np.array(control_points)
            point_map = {}

            def bezier(t):
                px = pts[:, 0] * np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
                py = pts[:, 1] * np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])
                point_map[int(np.sum(px))] = int(np.sum(py))

            for i in range(512):
                bezier((i / 511))

            for i in range(255):
                value_flags = image == i
                corrected_image[value_flags] = point_map[i]
            gc.collect()
        elif 'zcurve' in kwargs:
            points = self.get_kernel(**{'data': kwargs['zcurve']})
            corrected_image = image.copy()
            shadows_flags = image < int(points[0, 0])
            corrected_image[shadows_flags] = 0
            highlights_flags = image > int(points[0, 1])
            corrected_image[highlights_flags] = 255

        else:
            corrected_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        return corrected_image

    def arrayToImage(self, image):
        if not self.is_grayscale(image):
            b, g, r = cv2.split(image)
            image = cv2.merge((r, g, b))
        p = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=p)
        return photo

    def imageToBytes(self, image):
        b = io.BytesIO()
        image.save(b, 'gif')
        p = b.getvalue()
        photo = PhotoImage(data=p)
        return photo

    def crop(self, image, **kwargs):
        # For now it will only crop image in 1:1 ratio
        height, width, _ = image.shape
        h, w = height, width
        ratio = w / h  # W:H
        top, left = 0, 0
        right, bottom = 0, 0

        if w > h:
            w = h
        elif h > w:
            h = w

        if 'top' in kwargs:
            top = float(kwargs['top'])
            top = int(top * height)
            h += top
        if 'left' in kwargs:
            left = float(kwargs['left'])
            left = int(left * width)
            w += left
        '''
        if 'right' in kwargs:
            right = float(kwargs['right'])
            w -= int(right*width)
        if 'bottom' in kwargs:
            bottom = float(kwargs['bottom'])
            h -= int(bottom*width)
        '''

        return self.reframe(image[top:h, left:w])

    def reframe(self, image):
        if image is None:
            return np.zeros((360, 360, 3), np.uint8)
        h, w = image.shape[:2]
        h_w_ratio = h / w

        if w > h:
            h = int(h_w_ratio * 360)
            w = 360
        elif w < h:
            w = int(360 / h_w_ratio)
            h = 360
        else:
            w = h = 360
        image360 = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return image360

    def browseimg(self):
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        self.filename = askopenfilename()

        if len(self.filename) > 0:
            _, ext = os.path.splitext(self.filename)
            ext = str(ext).replace('.', '')
            image_formats = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg', 'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']
            video_formats = ['avi']

            if ext.lower() in image_formats:
                print(['Image Object', ext])
                img = cv2.imread(self.filename)
                self.originalImage = img
                # self.originalImage480x480 = cv2.resize(im, (480, 480), interpolation=cv2.INTER_AREA)
                self.originalImage360x360 = self.reframe(img)

                image = self.originalImage360x360
                photo = self.arrayToImage(image)
                self.originalImageView["image"] = photo
                self.originalPhoto = photo
                self.modifiedImageView["image"] = photo
                self.modifiedPhoto = photo
                self.applycommand()
            else:
                print(['Video Object', ext])
                self.video = cv2.VideoCapture(self.filename)
                self.video_playing = True
                playback = threading.Thread(group=None, target=self.playback, name=None, args=(), kwargs={})
                playback.start()

    def encrypt_command(self, data: str):
        while re.search(r'\[([^\[\]]*)(,)([^\[\]]*)\]', data):
            data = re.sub(r'\[([^\[\]]*)(,)([^\[\]]*)\]', r'[\1%44%\3]', data)
        return data

    def decrypt_command(self, data: str):
        data = data.replace('%44%', ',')
        return data

    def get_buffer(self, name: str):
        return self.buffers[name]

    def set_buffer(self, name: str, buffer):
        self.buffers[name] = buffer

    def clear_buffer(self):
        self.buffers = {}

    def get_kernel(self, **kwargs):
        kernel = None
        for key, value in kwargs.items():
            if 'name' in key:
                if value == 'buffer':
                    kernel = self.buffer_image
            elif 'type' in key:
                if 'zeros' in value:
                    size = int(str(value).replace('zeros', ''))
                    kernel = np.zeros((size, size), dtype=np.float32)
                elif 'ones' in value:
                    size = int(str(value).replace('ones', ''))
                    kernel = np.ones((size, size), dtype=np.float32)
                    kernel /= (size * size)

            elif 'data' in key:
                data = str(value).replace('%44%', ',').replace('[', '').replace(']', '')
                rows = data.split(';')
                kernel = []
                for row in rows:
                    row_data = row.split(',')
                    kernel.append(row_data)
                kernel = np.array(kernel, dtype=np.float32)
        if kernel is None:
            kernel = np.ones((5, 5), dtype=np.float32)
        return kernel

    def scale_move(self, event, element: Scale, name: str):
        if name.startswith('hsv'):
            h_low = self.h_low.get()
            s_low = self.s_low.get()
            v_low = self.v_low.get()
            h_high = self.h_high.get()
            s_high = self.s_high.get()
            v_high = self.v_high.get()
            hsv_lower = np.array([h_low, s_low, v_low])
            hsv_higher = np.array([h_high, s_high, v_high])

            hsv_image = cv2.cvtColor(self.originalImage360x360, cv2.COLOR_BGR2HSV)
            color_threshold_mask = cv2.inRange(hsv_image, hsv_lower, hsv_higher)
            segmented_image = cv2.bitwise_and(self.originalImage360x360,
                                              self.originalImage360x360,
                                              mask=color_threshold_mask
                                              )
            photo = self.arrayToImage(segmented_image)
            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo

    def stream(self):
        while self.cam_capturing:
            _, frame = self.cam.read()
            self.originalImage = frame
            self.originalImage360x360 = self.reframe(frame)
            image = self.originalImage360x360
            photo = self.arrayToImage(image)
            self.originalImageView["image"] = photo
            self.originalPhoto = photo
            self.applycommand(update=False)

        self.cam.release()
        self.cam = None

    def playback(self, **kwargs):
        while self.video_playing:
            _, frame = self.video.read()
            if frame is None:
                if self.video_playing is False:
                    break
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cv2.waitKey(34)
                continue

            self.originalImage = frame
            self.originalImage360x360 = self.reframe(frame)
            image = self.originalImage360x360
            photo = self.arrayToImage(image)
            self.originalImageView["image"] = photo
            self.originalPhoto = photo
            self.applycommand(update=False)
            cv2.waitKey(34)

    def webcam(self):
        if self.cam is None:
            self.cam = cv2.VideoCapture(0)

        if self.cam.isOpened():
            if self.cam_capturing:
                self.cam_capturing = False
            else:
                self.cam_capturing = True
                stream = threading.Thread(group=None, target=self.stream, name=None, args=(), kwargs={})
                stream.start()

    def update(self, mod_image):
        photo = self.arrayToImage(mod_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo

    def applycommand(self, update=True):
        '''options = self.optionsPanel.winfo_children()
        print(options)
        for option in options:
            print(option)
            opchilds = option.winfo_children()
            print(opchilds)
            for opchild in opchilds:
                print(type(opchild) is Scale)
                opchchs = opchild.winfo_children()
                print(opchchs)
            option.destroy()'''
        if update:
            self.commandText = self.command.get(1.0, END)
        inputcommand = self.commandText
        # print(self.commandText)
        commands = inputcommand.split("\n")
        # print len(commands)
        mod_image = self.originalImage360x360
        self.last_modified_image = mod_image
        line = 0
        for command in commands:
            line = line + 1
            if len(command) > 0:
                # print command
                if re.search(r'crop\((.*)\)', command, re.I):
                    m = re.search(r'crop\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    self.originalImage360x360 = self.crop(self.originalImage, **kwargs)
                    mod_image = self.originalImage360x360
                    self.last_modified_image = mod_image
                if command == "grayscale":
                    mod_image = self.grayscale(mod_image)
                    '''grayscale_ = Label(self.optionsPanel, text='Grayscale', padx=5, pady=5)
                    grayscale_.grid(row=1, column=1, sticky='w')'''
                elif re.search(r'quantize\((\d+)\)', command, re.I):
                    m = re.search(r'quantize\((\d+)\)', command, re.I)
                    mod_image = self.quantize(mod_image, int(float(m.group(1))))
                    '''quantize = Scale(self.optionsPanel, label="Quantize", from_=2, to=8,
                                     length=300, orient=HORIZONTAL)
                    quantize.grid(row=1, column=1, sticky='w')
                    quantize.set(int(float(m.group(1))))'''
                elif re.search(r'binary\((\d+)\)', command, re.I):
                    m = re.search(r'binary\((\d+)\)', command, re.I)
                    mod_image = self.binary(mod_image, int(float(m.group(1))))
                    '''binary = Scale(self.optionsPanel, label="Binary", from_=0, to=255,
                                   length=300, orient=HORIZONTAL)
                    binary.grid(row=2, column=1, sticky='w')
                    binary.set(int(float(m.group(1))))
                    binary._name = str(line) + '.0,' + str(line) + "." + str(len(command))
                    binary.bind("<B1-Motion>", lambda event, obj=binary: self.scale_move(event, obj))'''
                elif re.search(r'adaptive_threshold\((.*)\)', command, re.I):
                    m = re.search(r'adaptive_threshold\((.*)\)', command, re.I)
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.adaptive_threshold(mod_image, **kwargs)
                elif re.search(r'color_threshold\((.*)\)', command, re.I):
                    m = re.search(r'color_threshold\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.color_threshold(mod_image, **kwargs)
                elif re.search(r'rgb\((\w+)\)', command, re.I):
                    m = re.search(r'rgb\((\w+)\)', command, re.I)
                    mod_image = self.rgb(mod_image, m.group(1))
                elif re.search(r'extract_feature\((\d+)\)', command, re.I):
                    m = re.search(r'extract_feature\((\d+)\)', command, re.I)
                    mod_image = self.extract_feature(mod_image, int(float(m.group(1))))
                elif re.search(r'smooth\((\d+)\)', command, re.I):
                    m = re.search(r'smooth\((\d+)\)', command, re.I)
                    kwargs = {'ksize': int(float(m.group(1)))}
                    mod_image = self.smooth(mod_image, **kwargs)
                elif re.search(r'smooth\((.*)\)', command, re.I):
                    m = re.search(r'smooth\((.*)\)', command, re.I)
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.smooth(mod_image, **kwargs)
                elif re.search(r'blobs\((.*)\)', command, re.I):
                    m = re.search(r'blobs\((.*)\)', command, re.I)
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.blobs(mod_image, **kwargs)
                elif re.search(r'filter\((.*)\)', command, re.I):
                    m = re.search(r'filter\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.filter(mod_image, **kwargs)
                elif re.search(r'erosion\((.*)\)', command, re.I):
                    m = re.search(r'erosion\((.*)\)', command, re.I)
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.erosion(mod_image, **kwargs)
                elif re.search(r'dilatation\((.*)\)', command, re.I):
                    m = re.search(r'dilatation\((.*)\)', command, re.I)
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.dilatation(mod_image, **kwargs)
                elif re.search(r'morphology\((.*)\)', command, re.I):
                    m = re.search(r'morphology\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.morphology(mod_image, **kwargs)
                elif re.search(r'edges\((.*)\)', command, re.I):
                    m = re.search(r'edges\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.draw_edges(mod_image, **kwargs)
                elif re.search(r'convex_hull\((.*)\)', command, re.I):
                    m = re.search(r'convex_hull\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.draw_convex_hull(mod_image, **kwargs)
                elif re.search(r'bitwise\((.*)\)', command, re.I):
                    m = re.search(r'bitwise\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.bitwise(mod_image, **kwargs)
                elif re.search(r'color_adjustment\((.*)\)', command, re.I):
                    m = re.search(r'color_adjustment\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.color_adjustment(mod_image, **kwargs)
                elif re.search(r'save\((.*)\)', command, re.I):
                    m = re.search(r'save\((.*)\)', command, re.I)
                    self.set_buffer(m.group(1), mod_image)
                elif re.search(r'get\((.*)\)', command, re.I):
                    m = re.search(r'get\((.*)\)', command, re.I)
                    mod_image = self.get_buffer(m.group(1))
                elif re.search(r'sharpen\((.*)\)', command, re.I):
                    m = re.search(r'sharpen\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.sharpen(mod_image, **kwargs)
                elif re.search(r'play\((.*)\)', command, re.I):
                    m = re.search(r'play\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    if self.video_playing is False:
                        self.video_playing = True
                        playback = threading.Thread(group=None, target=self.playback, name=None, args=(), kwargs={})
                        playback.start()
                elif command == 'stop':
                    self.video_playing = False

        self.update(mod_image)
        self.last_modified_image = mod_image


root = Tk()
window = Window(root)
root.mainloop()
