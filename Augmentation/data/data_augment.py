import cv2
import numpy as np
import random

def distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-50, 50))

        #contrast distortion
        #if random.randrange(2):
        #    _convert(image, alpha=random.uniform(0.5, 1.5))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        #if random.randrange(2):
        #    _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        #if random.randrange(2):
        #    tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        #    tmp %= 180
        #    image[:, :, 0] = tmp

        #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-50, 50))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        #if random.randrange(2):
        #    _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        #if random.randrange(2):
        #    tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        #    tmp %= 180
        #    image[:, :, 0] = tmp

        #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        #if random.randrange(2):
        #    _convert(image, alpha=random.uniform(0.5, 1.5))

    return image
