import threading
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
from PIL import ImageTk
import io
import re
import os
import cv2
import numpy as np
import colorsys
import gc
from modules import *
from modules.features import Features
from modules.images import Images


class Window:
    def __init__(self, master):
        self.package = 'Image Processing Toolbox'
        master.title("Image Processing Toolbox")
        master.minsize(1200, 600)
        master.resizable(0, 0)
        self.master = master
        self.filename = ""
        self.commandText = ""
        self.features = {feature['name']: feature for feature in Features.collection}
        self.features_by_function = {feature['function_name']: feature for feature in Features.collection}

        # toolbar = Frame(root)
        # toolbar.grid(row=1, column=1)

        toolbar = LabelFrame(root)
        toolbar.grid(row=1, column=1)
        self.browseButton = Button(toolbar, text="Browse", command=self.browseimg)
        self.browseButton.grid(row=1, column=1, padx=2, pady=1)
        self.tasks = StringVar(root)

        options = [feature['name'] for feature in Features.collection if feature['menu_tree']]
        self.tasks.set(options[0])
        tasks = OptionMenu(*(toolbar, self.tasks) + tuple(options))
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
        # window = Toplevel(master)
        # im = Image.open("Default.png")
        im = cv2.imread('Default.png')
        self.originalImage = im
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
        # self.originalImage480x480 = cv2.resize(im, (480, 480), interpolation=cv2.INTER_AREA)
        Images.original_image_360 = cv2.resize(im, (360, 360), interpolation=cv2.INTER_AREA)
        Images.modified_image_360 = Images.original_image_360
        self.buffer_image = Images.modified_image_360
        self.buffers = {}
        self.cam = None  # cv2.VideoCapture(0)
        self.cam_capturing = False
        self.video = None
        self.video_playing = False

        image = Images.original_image_360
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
        task = self.tasks.get()
        append_at = self.command.index('end-1c')

        params = self.features[task]['parameters']
        defaults = [key + '= ' + str(val) for key, val in params.items()]
        task = task.replace(' ', '_')
        self.command.insert(append_at, task.lower() + '(' + ', '.join(defaults) + ')\n')

    def save_buffer(self):
        self.buffer_image = Images.modified_image_360

    def show_buffer(self):
        cv2.imshow('Buffer image', self.buffer_image)

    def is_grayscale(self, image):
        return len(image.shape) < 3

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

    def split(self, **kwargs):
        image = self.originalImage
        filename, ext = os.path.splitext(self.filename)
        rows = int(kwargs['rows']) if 'rows' in kwargs else 2
        columns = int(kwargs['columns']) if 'columns' in kwargs else 2
        grouped = kwargs['columns'] if 'columns' in kwargs else 'split'
        height, width = image.shape[:2]
        h, w = int(height / rows), int(width / columns)
        h0 = 0
        for r in range(rows):
            w0 = 0
            for c in range(columns):
                Features.set_buffer('split_' + str(r) + '_' + str(c),
                                    image[h0:np.clip(h0 + h, 0, height), w0:np.clip(w0 + w, 0, width)])
                w0 += w
            h0 += h

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
                Images.original_image_360 = self.reframe(img)

                image = Images.original_image_360
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

            hsv_image = cv2.cvtColor(Images.original_image_360, cv2.COLOR_BGR2HSV)
            color_threshold_mask = cv2.inRange(hsv_image, hsv_lower, hsv_higher)
            segmented_image = cv2.bitwise_and(Images.original_image_360,
                                              Images.original_image_360,
                                              mask=color_threshold_mask
                                              )
            Images.modified_image_360 = segmented_image
            photo = self.arrayToImage(segmented_image)
            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo

    def stream(self):
        while self.cam_capturing:
            _, frame = self.cam.read()
            self.originalImage = frame
            Images.original_image_360 = self.reframe(frame)
            image = Images.original_image_360
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
            Images.original_image_360 = self.reframe(frame)
            image = Images.original_image_360
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
        Images.modified_image_360 = mod_image
        photo = self.arrayToImage(mod_image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo

    def update_original(self, image):
        Images.original_image_360 = image
        photo = self.arrayToImage(image)
        self.originalImageView["image"] = photo
        self.originalPhoto = photo

    def applycommand(self, update=True):
        if update:
            self.commandText = self.command.get(1.0, END)
        inputcommand = self.commandText
        # print(self.commandText)
        commands = inputcommand.split("\n")
        # print len(commands)
        mod_image = Images.original_image_360
        line = 0
        for command in commands:
            line = line + 1
            if len(command) > 0:
                # print command
                m = re.search(r'([a-z0-9_]+)', command, re.I)
                function_name = m.group(1)
                if function_name in self.features_by_function:
                    feature = self.features_by_function[function_name]
                    pattern = re.compile(r'' + function_name + '\((.*)\)', re.I)
                    if re.search(pattern, command):
                        m = re.search(pattern, command)
                        command = self.encrypt_command(m.group(1))
                        command = command.replace(' ', '')
                        options = []
                        if not command == '':
                            options = command.split(',')
                        kwargs = {}
                        for option in options:
                            key, value = option.split('=')
                            kwargs[key] = value
                        callable_fun = feature['function']
                        mod_image = callable_fun(mod_image.copy(), **kwargs)
                elif re.search(r'crop\((.*)\)', command, re.I):
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
                    mod_image = Images.original_image_360 = self.crop(self.originalImage, **kwargs)
                elif re.search(r'split\((.*)\)', command, re.I):
                    m = re.search(r'split\((.*)\)', command, re.I)
                    command = self.encrypt_command(m.group(1))
                    command = command.replace(' ', '')
                    options = []
                    if not command == '':
                        options = command.split(',')
                    kwargs = {}
                    for option in options:
                        key, value = option.split('=')
                        kwargs[key] = value
                    self.split(**kwargs)
                elif re.search(r'save\((.*)\)', command, re.I):
                    m = re.search(r'save\((.*)\)', command, re.I)
                    Images.buffer[m.group(1)] = mod_image
                elif re.search(r'get\((.*)\)', command, re.I):
                    m = re.search(r'get\((.*)\)', command, re.I)
                    mod_image = Images.buffer[m.group(1)]
                elif re.search(r'set\((.*)\)', command, re.I):
                    m = re.search(r'set\((.*)\)', command, re.I)
                    im = Images.buffer[m.group(1)]
                    self.update_original(im)
                    self.update(im)
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
        Images.modified_image_360 = mod_image


root = Tk()
window = Window(root)
root.mainloop()
