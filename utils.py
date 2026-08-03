"""
utils.py
"""

import os
import ntpath
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

IMG_HEIGHT = 66
IMG_WIDTH = 200

LOG_COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

def _basename(path):
    return ntpath.basename(str(path).replace('\\', '/').replace('/', os.sep))


def load_log(csv_path):
   
    first = pd.read_csv(csv_path, nrows=1, header=None)
    has_header = any(str(v).strip().lower() in LOG_COLUMNS for v in first.iloc[0].tolist())
    if has_header:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(csv_path, header=None, names=LOG_COLUMNS)
    df['steering'] = pd.to_numeric(df['steering'], errors='coerce')
    df = df.dropna(subset=['steering']).reset_index(drop=True)
    return df


def resolve_image_path(raw_path, data_dir):
    
    raw_path = str(raw_path).strip()
    if os.path.isfile(raw_path):
        return raw_path

    rel = raw_path.replace('\\', '/').lstrip('/')
    cand = os.path.join(data_dir, *rel.split('/'))
    if os.path.isfile(cand):
        return cand

    name = _basename(raw_path)
    cand = os.path.join(data_dir, 'IMG', name)
    if os.path.isfile(cand):
        return cand

    for root, _dirs, files in os.walk(data_dir):
        if name in files:
            return os.path.join(root, name)
    return cand  


def load_dataset(csv_path, data_dir=None):
   
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(csv_path))
    df = load_log(csv_path)

    image_paths, steerings, missing = [], [], 0
    for center, steering in zip(df['center'], df['steering']):
        p = resolve_image_path(center, data_dir)
        if os.path.isfile(p):
            image_paths.append(p)
            steerings.append(float(steering))
        else:
            missing += 1
    if missing:
        print(f"[load_dataset] WARNING: {missing} image(s) referenced in the CSV "
              f"could not be found under {data_dir} and were skipped.")
    print(f"[load_dataset] usable samples: {len(image_paths)}")
    return np.array(image_paths), np.array(steerings, dtype=np.float32)


def balance_data(steerings, num_bins=25, samples_per_bin=400, seed=42):
    
    rng = np.random.default_rng(seed)
    edges = np.linspace(steerings.min(), steerings.max(), num_bins + 1)
    keep = []
    for b in range(num_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == num_bins - 1:
            idx = np.where((steerings >= lo) & (steerings <= hi))[0]
        else:
            idx = np.where((steerings >= lo) & (steerings < hi))[0]
        if len(idx) > samples_per_bin:
            idx = rng.choice(idx, samples_per_bin, replace=False)
        keep.extend(idx.tolist())
    keep = np.array(sorted(keep))
    print(f"[balance_data] {len(steerings)} -> {len(keep)} samples "
          f"({num_bins} bins, cap {samples_per_bin}/bin)")
    return keep


def plot_histogram(steerings, num_bins=25, title='Steering distribution',
                   samples_per_bin=None, out_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist, edges = np.histogram(steerings, num_bins)
    centers = (edges[:-1] + edges[1:]) * 0.5
    width = (edges[1] - edges[0]) * 0.9
    plt.figure(figsize=(8, 4))
    plt.bar(centers, hist, width=width)
    if samples_per_bin is not None:
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin), 'r--')
    plt.title(title)
    plt.xlabel('steering angle')
    plt.ylabel('count')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"[plot_histogram] saved {out_path}")
    plt.close()


def preprocess(img):
    
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img


def _random_pan(img):
    h, w = img.shape[:2]
    tx = np.random.uniform(-0.10, 0.10) * w
    ty = np.random.uniform(-0.10, 0.10) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _random_zoom(img):
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.5, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _random_rotation(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _random_flip(img, steering):
    img = cv2.flip(img, 1)
    return img, -steering


def random_augment(image_path, steering):

    img = mpimg.imread(image_path)          # RGB uint8
    if img.dtype != np.uint8:               # some readers return float 0..1
        img = (img * 255).astype(np.uint8)
    if np.random.rand() < 0.5:
        img = _random_pan(img)
    if np.random.rand() < 0.5:
        img = _random_zoom(img)
    if np.random.rand() < 0.5:
        img = _random_brightness(img)
    if np.random.rand() < 0.5:
        img = _random_rotation(img)
    if np.random.rand() < 0.5:
        img, steering = _random_flip(img, steering)
    return img, steering


def batch_generator(image_paths, steerings, batch_size, is_training):
   
    n = len(image_paths)
    while True:
        batch_img, batch_steer = [], []
        for _ in range(batch_size):
            i = random.randint(0, n - 1)
            if is_training:
                img, steering = random_augment(image_paths[i], steerings[i])
            else:
                img = mpimg.imread(image_paths[i])
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                steering = steerings[i]
            batch_img.append(preprocess(img))
            batch_steer.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steer)


def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Flatten(),
        Dense(1164, activation='elu'),
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model
