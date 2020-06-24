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
                           'Grayscale', 'Binary', 'Adaptive_Threshold',
                           'Color_Threshold',
                           'Smooth', 'Sharp', 'Blobs', 'Filter',
                           'Erosion', 'Dilatation'
                           )
        tasks.grid(row=1, column=2, padx=2, pady=1)
        self.addButton = Button(toolbar, text="Add", command=self.add_task)
        self.addButton.grid(row=1, column=3, padx=2, pady=1)
        self.chooseColorButton = Button(toolbar, text="Choose Color", command=self.choose_color)
        self.chooseColorButton.grid(row=1, column=4, padx=2, pady=1)


        # im = Image.open("Default.png")
        im = cv2.imread('Default.png')
        self.originalImage = im
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        # self.originalImage480x480 = cv2.resize(im, (480, 480), interpolation=cv2.INTER_AREA)
        self.originalImage360x360 = cv2.resize(im, (360, 360), interpolation=cv2.INTER_AREA)
        self.cvImage = None

        image = self.originalImage360x360
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

        '''self.legend = LabelFrame(optionsPanel, text='Grayscale', padx=5, pady=5)
        self.legend.grid(row=2, column=1)
        for i in range(1, 6):
            binaryScale = Scale(self.legend, label="Quantize " + str(i), from_=2, to=8, length=200,
                                orient=HORIZONTAL)
            binaryScale.grid(row=i, column=1)'''

    def rgb_to_hsv(self, r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        return h*360, s*100, v*100

    def choose_color(self):
        color = askcolor()
        r, g, b = color[0]
        hsv = self.rgb_to_hsv(r, g, b)
        print(hsv)

    def add_task(self):
        task = self.tasks.get()
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
        elif task == 'Sharp':
            self.command.insert(append_at, task.lower() + '(128)\n')
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

    def is_grayscale(self, image):
        return len(image.shape) < 3

    def grayscale(self, image):
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        photo = self.arrayToImage(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def quantize(self, image, quantizeValue):
        # bug: the whole programme is now using
        # an opencv image not PIL image object
        # Need to resolve
        image = image.quantize(quantizeValue)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def binary(self, image, threshold, thresh_type=cv2.THRESH_BINARY):
        # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(image, threshold, 255, thresh_type)
        photo = self.arrayToImage(threshold_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
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
        photo = self.arrayToImage(threshold_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
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
        photo = self.arrayToImage(blur_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return blur_image

    def blobs(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/d0/d7a/classcv_1_1SimpleBlobDetector.html
        # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        draw_color = (0, 255, 0)
        params = cv2.SimpleBlobDetector_Params()

        # keys:
        #       'minArea=100',
        #       'maxArea',
        #       'minCircularity=0.9',
        #       'maxCircularity',
        #       'minConvexity',
        #       'maxConvexity',
        #       'minInertia',
        #       'maxInertia',
        #       'color',
        #       'draw_color'
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

        photo = self.arrayToImage(blobs)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
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
        photo = self.arrayToImage(filter_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
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
        photo = self.arrayToImage(erosion_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return erosion_image

    def dilatation(self, image, **kwargs):
        # https://docs.opencv.org/4.2.0/db/df6/tutorial_erosion_dilatation.html
        dilatation_size = 3
        dilatation_type = cv2.MORPH_CROSS
        for key, value in kwargs.items():
            if 'size' in key:
                dilatation_size = int(float(value))
            if 'type' in key:
                if value == 'rect':
                    dilatation_type = cv2.MORPH_RECT
                elif value == 'cross':
                    dilatation_type = cv2.MORPH_CROSS
                elif value == 'ellipse':
                    dilatation_type = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))
        dilatation_image = cv2.dilate(image, element)
        photo = self.arrayToImage(dilatation_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
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
        photo = self.arrayToImage(color_threshold_mask)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return color_threshold_mask

    def contour(self, image):
        image = image.filter(ImageFilter.CONTOUR)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def detail(self, image):
        image = image.filter(ImageFilter.DETAIL)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def edge_enhance(self, image):
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def edge_enhance_more(self, image):
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def emboss(self, image):
        image = image.filter(ImageFilter.EMBOSS)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def find_edges(self, image):
        image = image.filter(ImageFilter.FIND_EDGES)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def sharpen(self, image):
        image = image.filter(ImageFilter.SHARPEN)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

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

    def browseimg(self):
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        self.filename = askopenfilename()

        if len(self.filename) > 0:
            img = cv2.imread(self.filename)
            self.originalImage = img

            # self.originalImage480x480 = cv2.resize(im, (480, 480), interpolation=cv2.INTER_AREA)
            self.originalImage360x360 = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)

            image = self.originalImage360x360
            photo = self.arrayToImage(image)
            self.originalImageView["image"] = photo
            self.originalPhoto = photo
            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo
            self.applycommand()

    def encrypt_command(self, data: str):
        while re.search(r'\[([^\[\]]*)(,)([^\[\]]*)\]', data):
            data = re.sub(r'\[([^\[\]]*)(,)([^\[\]]*)\]', r'[\1%44%\3]', data)
        return data

    def decrypt_command(self, data: str):
        data = data.replace('%44%', ',')
        return data

    def get_kernel(self, **kwargs):
        kernel = None
        for key, value in kwargs.items():
            if 'name' in key:
                # get popular kernel
                pass
            elif 'type' in key:
                if 'zeros' in value:
                    size = int(str(value).replace('zeros', ''))
                    kernel = np.zeros((size, size), dtype=np.float32)
                elif 'ones' in value:
                    size = int(str(value).replace('ones', ''))
                    kernel = np.ones((size, size), dtype=np.float32)
                    kernel /= (size*size)
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

    def scale_move(self, event, element: Scale):
        print(element.get())
        print(element.name)

    def applycommand(self):
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

        self.commandText = self.command.get(1.0, END)
        inputcommand = self.commandText
        # print(self.commandText)
        commands = inputcommand.split("\n")
        # print len(commands)
        image = self.originalImage360x360
        mod_image = self.originalImage360x360
        line = 0
        for command in commands:
            line = line + 1
            if len(command) > 0:
                # print command
                if command == "grayscale":
                    mod_image = self.grayscale(mod_image)
                    grayscale_ = Label(self.optionsPanel, text='Grayscale', padx=5, pady=5)
                    grayscale_.grid(row=1, column=1, sticky='w')
                elif re.search(r'quantize\((\d+)\)', command, re.I):
                    m = re.search(r'quantize\((\d+)\)', command, re.I)
                    mod_image = self.quantize(mod_image, int(float(m.group(1))))
                    quantize = Scale(self.optionsPanel, label="Quantize", from_=2, to=8,
                                     length=300, orient=HORIZONTAL)
                    quantize.grid(row=1, column=1, sticky='w')
                    quantize.set(int(float(m.group(1))))
                elif re.search(r'binary\((\d+)\)', command, re.I):
                    m = re.search(r'binary\((\d+)\)', command, re.I)
                    mod_image = self.binary(mod_image, int(float(m.group(1))))
                    binary = Scale(self.optionsPanel, label="Binary", from_=0, to=255,
                                   length=300, orient=HORIZONTAL)
                    binary.grid(row=2, column=1, sticky='w')
                    binary.set(int(float(m.group(1))))
                    binary._name = str(line) + '.0,' + str(line) + "." + str(len(command))
                    binary.bind("<B1-Motion>", lambda event, obj=binary: self.scale_move(event, obj))
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
                elif command == "contour":
                    mod_image = self.contour(mod_image)
                elif command == "detail":
                    mod_image = self.detail(mod_image)
                elif command == "edge_enhance":
                    mod_image = self.edge_enhance(mod_image)
                elif command == "edge_enhance_more":
                    mod_image = self.edge_enhance_more(mod_image)
                elif command == "emboss":
                    mod_image = self.emboss(mod_image)
                elif command == "find_edges":
                    mod_image = self.find_edges(mod_image)
                elif command == "sharpen":
                    mod_image = self.sharpen(mod_image)


root = Tk()

window = Window(root)

root.mainloop()
# root.destroy()
