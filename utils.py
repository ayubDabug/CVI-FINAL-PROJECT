import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(DATA_DIR, "IMG")
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")

columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
data = pd.read_csv(CSV_PATH, names=columns)

def path_leaf(path):
    return os.path.basename(path.strip())


data['Center'] = data['Center'].apply(path_leaf)
print(f"Total samples: {len(data)}")

def zoom(image):
    scale = random.uniform(1.0, 1.3)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
    return resized[start_y:start_y + h, start_x:start_x + w]


def panning(image):
    h, w = image.shape[:2]
    tx, ty = random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))


def brightness(image):
    factor = random.uniform(0.4, 1.3)
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def rotation(image):
    h, w = image.shape[:2]
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def flip(image, steering_angle):
    return cv2.flip(image, 1), -steering_angle


def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = panning(image)
    if np.random.rand() < 0.5:
        image = brightness(image)
    if np.random.rand() < 0.5:
        image = rotation(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip(image, steering_angle)
    return image, steering_angle

# PREPROCESSING
def preprocess_image(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    return image / 255

# DATA
image_paths = data['Center'].values
steering_angles = data['Steering'].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, steering_angles
)

# LOAD & PREPROCESS EVERYTHING INTO MEMORY
def load_and_process(paths, steering_angles, augment=False):
    images = []
    steerings = []
    total = len(paths)
    for i, (path, steer) in enumerate(zip(paths, steering_angles)):
        full_path = os.path.join(IMAGE_FOLDER, path)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 loads as BGR, convert to RGB

        if augment:
            image, steer = random_augment(image, steer)

        image = preprocess_image(image)
        images.append(image)
        steerings.append(steer)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} images")

    return np.asarray(images), np.asarray(steerings)

print("\nLoading and preprocessing training images (this may take a minute)...")
X_train, y_train = load_and_process(X_train, y_train, augment=True)

X_test, y_test = load_and_process(X_test, y_test, augment=False)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# For batching prof said you can just add batch_size in model.fit()


# SAVE THE MODEL
# model.save('model.h5')
