"""
Loads driving_log.csv and reviews/balances the steering angle distribution.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 3. Loading the driving log
# ---------------------------------------------------------------------------
def get_name(file_path):
    """Return just the filename, independent of the machine that recorded it."""
    return os.path.basename(file_path.strip())


def load_data(data_dir, csv_name="driving_log.csv"):
    """Read driving_log.csv and return absolute center-image paths + steering angles."""
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    csv_path = os.path.join(data_dir, csv_name)
    data = pd.read_csv(csv_path, names=columns)

    data["center"] = data["center"].apply(get_name)

    image_paths = data["center"].apply(lambda f: os.path.join(data_dir, "IMG", f)).values
    steerings = data["steering"].values.astype(np.float32)

    print(f"Loaded {len(image_paths)} samples from {csv_path}")
    return image_paths, steerings


# ---------------------------------------------------------------------------
# 4. Reviewing and balancing the dataset
# ---------------------------------------------------------------------------
def balance_data(image_paths, steerings, num_bins=25, samples_per_bin=400, plot=True, save_dir=None):
    """Flatten the steering histogram so near-zero angles don't dominate training."""
    hist, bins = np.histogram(steerings, num_bins)

    if plot:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.figure(figsize=(8, 4))
        plt.bar(center, hist, width=0.05)
        plt.plot((np.min(steerings), np.max(steerings)),
                 (samples_per_bin, samples_per_bin), "r-")
        plt.title("Steering angle distribution (before balancing)")
        plt.xlabel("Steering angle")
        plt.ylabel("Number of samples")
        if save_dir:
            plt.savefig(os.path.join(save_dir, "histogram_before.png"))
        plt.close()

    remove_idx = []
    for j in range(num_bins):
        bin_idx = [i for i in range(len(steerings))
                   if bins[j] <= steerings[i] <= bins[j + 1]]
        bin_idx = list(np.random.permutation(bin_idx))
        remove_idx.extend(bin_idx[samples_per_bin:])

    print(f"Removed {len(remove_idx)} over-represented samples, "
          f"{len(steerings) - len(remove_idx)} remaining")

    image_paths = np.delete(image_paths, remove_idx, axis=0)
    steerings = np.delete(steerings, remove_idx, axis=0)

    if plot:
        hist, _ = np.histogram(steerings, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.figure(figsize=(8, 4))
        plt.bar(center, hist, width=0.05)
        plt.plot((np.min(steerings), np.max(steerings)),
                 (samples_per_bin, samples_per_bin), "r-")
        plt.title("Steering angle distribution (after balancing)")
        plt.xlabel("Steering angle")
        plt.ylabel("Number of samples")
        if save_dir:
            plt.savefig(os.path.join(save_dir, "histogram_after.png"))
        plt.close()

    return image_paths, steerings
