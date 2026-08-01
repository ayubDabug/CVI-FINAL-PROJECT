"""
Data augmentation — applied only to the training split, and only to a random
subset of samples (never uniformly across the whole dataset).
"""
import cv2
import numpy as np


def pan(img):
    tx = np.random.uniform(-0.1, 0.1) * img.shape[1]
    ty = np.random.uniform(-0.1, 0.1) * img.shape[0]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def zoom(img):
    scale = np.random.uniform(1.0, 1.3)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, M, (w, h))


def img_random_brightness(img):
    factor = np.random.uniform(0.4, 1.3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def rotate(img, max_angle=8):
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def img_random_flip(img, steering_angle):
    img = cv2.flip(img, 1)
    steering_angle = -steering_angle
    return img, steering_angle


def random_augment(image_path, steering_angle):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Each transform is applied to only a random subset of samples.
    if np.random.rand() < 0.5:
        img = pan(img)
    if np.random.rand() < 0.5:
        img = zoom(img)
    if np.random.rand() < 0.5:
        img = img_random_brightness(img)
    if np.random.rand() < 0.5:
        img = rotate(img)
    if np.random.rand() < 0.5:
        img, steering_angle = img_random_flip(img, steering_angle)

    return img, steering_angle
