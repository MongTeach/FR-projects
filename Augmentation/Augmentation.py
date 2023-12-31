import glob
import cv2
import os
from PIL import Image
import numpy as np
import random

i = 0
clas = ['dion']
ext = ".jpg"
dire = "Result"
source = "dataset/"
cls = []

def _convert(image, alpha=1, beta=0):
    tmp = image.astype(float) * alpha + beta
    tmp[tmp < 0] = 0
    tmp[tmp > 255] = 255
    image[:] = tmp

# Creating directories
try:
    os.mkdir(dire)
    print("Directory", dire, "created")
except FileExistsError:
    print("Directory", dire, "already exists")

for cls in clas:
    path = os.path.join(dire, cls)
    try:
        os.mkdir(path)
        print("Directory", path, "created")
    except FileExistsError:
        print("Directory", path, "already exists")

for cls in clas:
    locate = os.path.join(source, cls, "*" + ext)
    for gt_file in glob.glob(locate):
        name = os.path.basename(gt_file)
        print(name)
        i += 1
        im = Image.open(gt_file)
        im.save(os.path.join(dire, cls, name))

        im = cv2.imread(gt_file, cv2.IMREAD_COLOR)

        # Performing operations on the image and saving it with different variations
        
        # Brightness variation
        _convert(im, beta=random.uniform(-30, 50))
        cv2.imwrite(os.path.join(dire, cls, "Bright_rand_" + name), im)

        # Contrast variation
        _convert(im, alpha=random.uniform(0.7, 1.5))
        cv2.imwrite(os.path.join(dire, cls, "Bright_Contrast_rand_" + name), im)

        # Saturation variation
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        _convert(im_hsv[:, :, 1], alpha=random.uniform(0.5, 1.5))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(dire, cls, "Bright_Contrast_Sat_rand_" + name), im)

        # Hue variation
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        tmp = im_hsv[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        im_hsv[:, :, 0] = tmp
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(dire, cls, "Bright_Contrast_Sat_Hue_rand_" + name), im)

        # Rotation
        h, w = im.shape[:2]
        angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        im = cv2.warpAffine(im, M, (w, h))
        cv2.imwrite(os.path.join(dire, cls, "Rot_rand_" + name), im)

        # Flip horizontal
        im = cv2.flip(im, 1)
        cv2.imwrite(os.path.join(dire, cls, "Flip_rand_" + name), im)
