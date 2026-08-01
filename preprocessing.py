"""
Image preprocessing applied identically to every image — training, validation,
and live inference in the simulator.
"""
import cv2

IMG_WIDTH = 200
IMG_HEIGHT = 66


def img_preprocess(img):
    img = img[60:135, :, :]                       # crop out sky / hood -> road area
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     # Nvidia model expects YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0                              # normalize
    return img
