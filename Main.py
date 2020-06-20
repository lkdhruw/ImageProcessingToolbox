from tkinter import *
from PIL import Image
from PIL import ImageFilter
from PIL import ImageTk
import io
import re
import cv2
import binascii
import numpy as np


class Window:
    def __init__(self, master):
        master.title("Digital Image Processing")
        master.minsize(800, 600)

        self.filename = ""
        self.commandText = ""

        # toolbar = Frame(root)
        # toolbar.grid(row=1, column=1)

        self.browseButton = Button(root, text="Browse", command=self.browseimg)
        self.browseButton.grid(row=1, column=1)

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

        self.command = Text(root, height=10, width=80)
        self.command.grid(row=4, column=1, columnspan=2, pady=(10, 10))

        self.applyButton = Button(root, text="Apply", command=self.applycommand)
        self.applyButton.grid(row=5, column=2)

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
        # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        if not self.is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100

        params.filterByCircularity = True
        params.minCircularity = 0.9

        params.filterByConvexity = True
        params.minConvexity = 0.2

        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        photo = self.arrayToImage(blobs)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return blobs

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

    def applycommand(self):
        self.commandText = self.command.get(1.0, END)
        inputcommand = self.commandText
        # print inputCommand
        commands = inputcommand.split("\n")
        # print len(commands)
        image = self.originalImage360x360
        mod_image = self.originalImage360x360
        for command in commands:
            if len(command) > 0:
                # print command
                if command == "grayscale":
                    mod_image = self.grayscale(mod_image)
                elif re.search(r'quantize\((\d+)\)', command, re.I):
                    m = re.search(r'quantize\((\d+)\)', command, re.I)
                    mod_image = self.quantize(mod_image, int(float(m.group(1))))
                elif re.search(r'binary\((\d+)\)', command, re.I):
                    m = re.search(r'binary\((\d+)\)', command, re.I)
                    mod_image = self.binary(mod_image, int(float(m.group(1))))
                elif re.search(r'adaptive_threshold\((.*)\)', command, re.I):
                    m = re.search(r'adaptive_threshold\((.*)\)', command, re.I)
                    options = ((m.group(1)).replace(' ', '')).split(',')
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
