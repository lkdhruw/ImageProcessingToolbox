from tkinter import *
from PIL import Image
from PIL import ImageFilter
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

        #toolbar = Frame(root)
        #toolbar.grid(row=1, column=1)

        self.browseButton = Button(root, text="Browse", command=self.browseimg)
        self.browseButton.grid(row=1, column=1)

        #im = Image.open("lena.png")
        im = Image.open("Default.png")
        self.originalImage = im

        #self.originalImage480x480 = self.originalImage
        #self.originalImage480x480.thumbnail((480, 480))
        self.originalImage360x360 = self.originalImage
        self.originalImage360x360.thumbnail((360, 360))
        #self.originalImage240x240 = self.originalImage
        #self.originalImage240x240.thumbnail((240, 240))
        self.cvImage = None

        image = self.originalImage360x360
        photo = self.imageToBytes(image)
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

    def grayscale(self,image):
        #image = self.originalImage360x360
        image = image.convert("L")
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def quantize(self, image, quantizeValue):
        #image = self.originalImage360x360
        image = image.quantize(quantizeValue)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def binary(self, image, threshold):
        #image = self.originalImage360x360
        image = image.convert("L")
        width, _ = image.size
        black = 1
        white = 1
        for i, px in enumerate(image.getdata()):
            y = int(i / width)
            x = int(i % width)
            if px > threshold:
                image.putpixel((x, y), 255)
                white += 1
            else:
                image.putpixel((x, y), 0)
                black += 1
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

    def extract_feature(self, originalImage, threshold):
        image = originalImage.copy()
        binImage = self.binary(image,threshold)
        if image.mode == "RGB":
            width, _ = image.size
            for i, px in enumerate(binImage.getdata()):
                y = int(i / width)
                x = int(i % width)
                if px == 0:
                    image.putpixel((x, y), (px, 0, 0))

            photo = self.imageToBytes(image)

            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo

        return image

    def rgb(self, originalImage, options):
        image = originalImage.copy()
        if image.mode == "RGB":
            R, G, B = image.split()
            width, _ = image.size
            listR = list(R.getdata())
            listG = list(G.getdata())
            listB = list(B.getdata())

            if options == "r":
                for i, px in enumerate(R.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (px, 0, 0))
                photo = self.imageToBytes(image)
            elif options == "g":
                for i, px in enumerate(G.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (0, px, 0))
                photo = self.imageToBytes(image)
            elif options == "b":
                for i, px in enumerate(B.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (0, 0, px))
                photo = self.imageToBytes(image)
            elif options == "gb":
                for i, px in enumerate(G.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (0, px, listB[i]))
                photo = self.imageToBytes(image)
            elif options == "rb":
                for i, px in enumerate(B.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (listR[i], 0, px))
                photo = self.imageToBytes(image)
            elif options == "rg":
                for i, px in enumerate(R.getdata()):
                    y = int(i / width)
                    x = int(i % width)
                    image.putpixel((x, y), (px, listG[i], 0))
                photo = self.imageToBytes(image)
            else:
                photo = self.imageToBytes(image)

            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo

        return image

    def blur(self, image):
        image = image.filter(ImageFilter.BLUR)
        photo = self.imageToBytes(image)
        self.modifiedImageView["image"] = photo
        self.modifiedPhoto = photo
        return image

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

    def imageToBytes(self,image):
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
            self.originalImage = Image.open(self.filename)
            pil_image = self.originalImage.copy()
            open_cv_image = np.array(pil_image.convert('RGB'))
            self.cvImage = open_cv_image[:, :, ::-1].copy() # cv2.imread(self.filename, 0)
            #self.originalImage480x480 = self.originalImage
            #self.originalImage480x480.thumbnail((480, 480))
            self.originalImage360x360 = self.originalImage
            self.originalImage360x360.thumbnail((360, 360))
            #self.originalImage240x240 = self.originalImage
            #self.originalImage240x240.thumbnail((240, 240))

            image = self.originalImage360x360
            photo = self.imageToBytes(image)
            self.originalImageView["image"] = photo
            self.originalPhoto = photo
            self.modifiedImageView["image"] = photo
            self.modifiedPhoto = photo

    def applycommand(self):
        self.commandText = self.command.get(1.0,END)
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
                    mod_image = self.quantize(mod_image,int(float(m.group(1))))
                elif re.search(r'binary\((\d+)\)', command, re.I):
                    m = re.search(r'binary\((\d+)\)', command, re.I)
                    mod_image = self.binary(mod_image,int(float(m.group(1))))
                elif re.search(r'rgb\((\w+)\)', command, re.I):
                    m = re.search(r'rgb\((\w+)\)', command, re.I)
                    mod_image = self.rgb(mod_image,m.group(1))
                elif re.search(r'extract_feature\((\d+)\)', command, re.I):
                    m = re.search(r'extract_feature\((\d+)\)', command, re.I)
                    mod_image = self.extract_feature(mod_image,int(float(m.group(1))))
                elif command == "blur":
                    mod_image = self.blur(mod_image)
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
#root.destroy()
