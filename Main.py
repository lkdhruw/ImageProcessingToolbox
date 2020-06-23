from tkinter import *
from PIL import Image
from PIL import ImageFilter
from PIL import ImageTk
import io
import re
import cv2
import binascii
import numpy as np
import time


class Binary:
    def __init__(self, value: int):
        self.value = value

    def command(self):
        return 'binary(' + str(self.value) + ')'


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
                           'Smooth', 'Sharp', 'Blobs', 'Filter'
                           )
        tasks.grid(row=1, column=2, padx=2, pady=1)
        self.addButton = Button(toolbar, text="Add", command=self.add_task)
        self.addButton.grid(row=1, column=3, padx=2, pady=1)

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
        self.command.bind('<Key>', command_edit)

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

    def add_task(self):
        task = self.tasks.get()
        append_at = self.command.index('end-1c')
        if task == 'Grayscale':
            self.command.insert(append_at, task.lower()+'\n')
        elif task == 'Binary':
            self.command.insert(append_at, task.lower()+'(128)\n')
        elif task == 'Adaptive_Threshold':
            self.command.insert(append_at, task.lower()+'(128)\n')
        elif task == 'Smooth':
            self.command.insert(append_at, task.lower()+'(128)\n')
        elif task == 'Sharp':
            self.command.insert(append_at, task.lower()+'(128)\n')
        elif task == 'Blobs':
            self.command.insert(append_at, task.lower()+'(128)\n')
        elif task == 'Filter':
            self.command.insert(append_at, task.lower()+'(128)\n')

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

    def adaptive_threshold(self, image, thresh_type=cv2.THRESH_BINARY, **kwargs):
        # https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        block_size = 11
        constant = 2
        for key, value in kwargs.items():
            if key == 'adaptive_method':
                # cv.ADAPTIVE_THRESH_MEAN_C
                # cv.ADAPTIVE_THRESH_GAUSSIAN_C
                if value == 'MEAN':
                    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
                elif value == 'GAUSSIAN':
                    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            elif key == 'block_size':
                block_size = int(float(value))
            elif key == 'constant':
                constant = int(float(value))

        threshold_image = cv2.adaptiveThreshold(image, 255,
                                                adaptive_method, thresh_type,
                                                block_size, constant)
        photo = self.arrayToImage(threshold_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return threshold_image

    def blur(self, image, ksize):
        # https://docs.opencv.org/4.2.0/d4/d13/tutorial_py_filtering.html
        # Need implement more filters such as gaussian, median etc
        blur_image = cv2.blur(image, (ksize, ksize))
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

        photo = self.arrayToImage(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return photo

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

    def smooth(self, image):
        image = image.filter(ImageFilter.SMOOTH)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def smooth_more(self, image):
        image = image.filter(ImageFilter.SMOOTH_MORE)
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
                elif re.search(r'rgb\((\w+)\)', command, re.I):
                    m = re.search(r'rgb\((\w+)\)', command, re.I)
                    mod_image = self.rgb(mod_image, m.group(1))
                elif re.search(r'extract_feature\((\d+)\)', command, re.I):
                    m = re.search(r'extract_feature\((\d+)\)', command, re.I)
                    mod_image = self.extract_feature(mod_image, int(float(m.group(1))))
                elif re.search(r'blur\((\d+)\)', command, re.I):
                    m = re.search(r'blur\((\d+)\)', command, re.I)
                    mod_image = self.blur(mod_image, int(float(m.group(1))))
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
                    command = (m.group(1)).replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    mod_image = self.filter(mod_image, **kwargs)
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
                elif command == "smooth":
                    mod_image = self.smooth(mod_image)
                elif command == "smooth_more":
                    mod_image = self.smooth_more(mod_image)
                elif command == "sharpen":
                    mod_image = self.sharpen(mod_image)


root = Tk()

window = Window(root)

root.mainloop()
# root.destroy()
