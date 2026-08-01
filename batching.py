"""
Splits the dataset into batches for training/validation, applying
augmentation + preprocessing to each sample as it's pulled into a batch.
"""
import cv2
import numpy as np

from augmentation import random_augment
from preprocessing import img_preprocess


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(image_paths))

            if is_training:
                img, steering = random_augment(image_paths[idx], steering_ang[idx])
            else:
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steering_ang[idx]

            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)

        yield np.asarray(batch_img), np.asarray(batch_steering)
